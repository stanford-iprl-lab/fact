import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    
    ft_window: int = 27         # history window length shared by FT and state TCN encoders
    ft_out_dim: int = 6         # FT vector dimension per timestep (6 for 6-axis sensor)
    ft_tcn_hidden: int = 64     # TCN internal hidden channels
    ft_history_steps: int = 0   # 0 = disabled; > 0 = number of past steps to encode

    use_ln_scheduler: bool = False   # use LN instead of Beta
    ln_alpha: float = 1.5
    ln_sigma: float = 1.0

   
    force_adarms: bool = False
    force_adarms_cg_threshold: float = 0.0
    force_ft_norm_q01: tuple[float, ...] = ()
    force_ft_norm_q99: tuple[float, ...] = ()

    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.force_adarms and not self.pi05:
            raise ValueError("force_adarms requires pi05=True.")
        if self.force_adarms and self.ft_history_steps <= 0:
            raise ValueError("force_adarms requires ft_history_steps > 0.")
        if self.force_adarms and self.ft_out_dim < 3:
            raise ValueError("force_adarms requires ft_out_dim >= 3 for the 3D force contact gate.")
        if self.ft_history_steps < 0:
            raise ValueError("ft_history_steps must be non-negative.")
        if bool(self.force_ft_norm_q01) != bool(self.force_ft_norm_q99):
            raise ValueError("force_ft_norm_q01 and force_ft_norm_q99 must be provided together.")
        if self.force_ft_norm_q01:
            if len(self.force_ft_norm_q01) != self.ft_out_dim or len(self.force_ft_norm_q99) != self.ft_out_dim:
                raise ValueError(
                    "force_ft_norm_q01 and force_ft_norm_q99 must each have length ft_out_dim "
                    f"({self.ft_out_dim})."
                )
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    def ft_dim(self) -> int:
        """Flattened FT input dimension for the current burst; 0 when force input is disabled."""
        return self.ft_window * self.ft_out_dim if self.force_adarms else 0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                ft_current=jax.ShapeDtypeStruct([batch_size, self.ft_dim], jnp.float32) if self.ft_dim > 0 else None,
                ft_history=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.ft_history_steps * self.ft_window * self.ft_out_dim], jnp.float32
                    )
                    if self.ft_history_steps > 0
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
