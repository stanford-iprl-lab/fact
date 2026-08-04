import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def ln_sample(
    rng: at.KeyArrayLike,
    shape: tuple,
    alpha: float = 1.5,
    sigma: float = 1.0,
) -> at.Float[at.Array, "..."]:
    """Logit-Normal (LN) noise time sampler.

    Args:
        rng: JAX random key.
        shape: Output shape tuple.
        alpha: Logit-normal shift.
        sigma: Standard deviation of the logit-normal.
    Returns:
        Float array of shape ``shape`` with values in (0.001, 0.999).
    """
    z = jax.random.normal(rng, shape)
    return jnp.clip(jax.nn.sigmoid(sigma * z - alpha), 0.001, 0.999)

class CausalDilatedBlock(nnx.Module):
    """Single causal dilated 1-D conv block with layer-norm and residual connection.

    Input / output shape: ``(B, T, hidden)``.
    Causal masking is enforced by left-only padding of ``(kernel_size-1)*dilation``
    steps; the right side is never padded.
    """

    def __init__(self, hidden: int, kernel_size: int, dilation: int, rngs: nnx.Rngs):
        left_pad = (kernel_size - 1) * dilation
        self.conv = nnx.Conv(
            hidden,
            hidden,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding=((left_pad, 0),),
            kernel_dilation=(dilation,),
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(hidden, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:  # (B, T, C) → (B, T, C)
        return x + nnx.swish(self.norm(self.conv(x)))


class TCNEncoder(nnx.Module):
    """Temporal convolutional network (TCN) for history sequences.

    Encodes a flattened window ``(B, window * in_dim)`` into one final-summary
    token ``(B, 1, out_dim)`` using a stack of causal dilated 1-D conv blocks.

    """

    def __init__(
        self,
        in_dim: int,
        window: int,
        hidden: int,
        out_dim: int,
        rngs: nnx.Rngs,
    ):
        self.in_dim = in_dim
        self.window = window
        self.input_proj = nnx.Conv(in_dim, hidden, kernel_size=(1,), rngs=rngs)
        self.block0 = CausalDilatedBlock(hidden, kernel_size=3, dilation=1, rngs=rngs)
        self.block1 = CausalDilatedBlock(hidden, kernel_size=3, dilation=2, rngs=rngs)
        self.block2 = CausalDilatedBlock(hidden, kernel_size=3, dilation=4, rngs=rngs)
        self.block3 = CausalDilatedBlock(hidden, kernel_size=3, dilation=8, rngs=rngs)
        self.output_proj = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x_flat: jax.Array) -> jax.Array:
        """
        Args:
            x_flat: ``(B, window * in_dim)`` flattened history sequence.
        Returns:
            ``(B, 1, out_dim)`` final-summary token array.
        """
        B = x_flat.shape[0]
        x = x_flat.reshape(B, self.window, self.in_dim)  # (B, T, in_dim)
        x = self.input_proj(x)                              # (B, T, hidden)
        for block in [self.block0, self.block1, self.block2, self.block3]:
            x = block(x)
        x = x[:, -1:, :]                                     # (B, 1, hidden)
        return self.output_proj(x)                            # (B, 1, out_dim)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_ln_scheduler = config.use_ln_scheduler
        self.ln_alpha = config.ln_alpha
        self.ln_sigma = config.ln_sigma
        self.ft_out_dim = config.ft_out_dim
        self.force_adarms = config.force_adarms
        self._force_adarms_cg_threshold = config.force_adarms_cg_threshold
        # Keep normalization statistics as static, hashable graph metadata.
        self._force_ft_norm_q01 = config.force_ft_norm_q01 or None
        self._force_ft_norm_q99 = config.force_ft_norm_q99 or None
        self.ft_history_steps = config.ft_history_steps

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # FT history encoder
        if config.ft_history_steps > 0:
            self.ft_tcn_encoder = TCNEncoder(
                in_dim=config.ft_out_dim,
                window=config.ft_window,
                hidden=config.ft_tcn_hidden,
                out_dim=action_expert_config.width,
                rngs=rngs,
            )

        # Force adaRMS encoder
        # Mean-pools the current burst then projects to adaRMS space.
        if config.force_adarms:
            self.force_adarms_mlp_in = nnx.Linear(config.ft_out_dim, action_expert_config.width, rngs=rngs)
            self.force_adarms_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.force_gamma_proj = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | dict | None,
    ]:
        """Embed state, historical FT tokens, and noisy action tokens.

        The current FT burst conditions Pi0.5 through adaRMS and is not appended
        to the suffix token sequence.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # Add a single state token.
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        if self.ft_history_steps > 0 and obs.ft_history is not None:
            B = obs.ft_history.shape[0]
            # Reshape to (B * ft_history_steps, ft_window * ft_out_dim) for batched TCN
            step_flat = obs.ft_history.reshape(B * self.ft_history_steps, -1)
            step_tokens = self.ft_tcn_encoder(step_flat).reshape(B, self.ft_history_steps, -1)
            tokens.append(step_tokens)
            input_mask.append(jnp.ones((B, self.ft_history_steps), dtype=jnp.bool_))
            # Start a new ar-block for history tokens (they can all attend to each other)
            ar_mask += [True] + [False] * (self.ft_history_steps - 1)

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

            ft_gamma = None
            if self.force_adarms and obs.ft_current is not None:
                B = obs.ft_current.shape[0]
                ft_mean = obs.ft_current.reshape(B, -1, self.ft_out_dim).mean(axis=1)  # (B, ft_out_dim)
                ft_emb = self.force_adarms_mlp_in(ft_mean)
                ft_emb = nnx.swish(ft_emb)
                ft_emb = self.force_adarms_mlp_out(ft_emb)
                ft_gamma = self.force_gamma_proj(ft_emb)
                if self._force_adarms_cg_threshold > 0.0:
                    # Undo quantile normalization to recover raw sensor units.
                    ft_for_threshold = ft_mean
                    if self._force_ft_norm_q01 is not None:
                        q01 = jnp.asarray(self._force_ft_norm_q01, dtype=ft_mean.dtype)
                        q99 = jnp.asarray(self._force_ft_norm_q99, dtype=ft_mean.dtype)
                        ft_for_threshold = (ft_mean + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                    # Contact is determined only by the 3D force vector. Torque
                    # channels still condition adaRMS but do not affect the gate.
                    force_mag = jnp.linalg.norm(ft_for_threshold[..., :3], axis=-1, keepdims=True)  # (B, 1)
                    is_contact = force_mag > self._force_adarms_cg_threshold                  # (B, 1)
                    ft_gamma = jnp.where(is_contact, ft_gamma, jnp.zeros_like(ft_gamma))
            action_expert_tokens = action_tokens
            if self.force_adarms:
                adarms_cond = {"time": time_emb, "force_gamma": ft_gamma}
            else:
                adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if self.use_ln_scheduler:
            time = ln_sample(time_rng, batch_shape, self.ln_alpha, self.ln_sigma)
        else:
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
