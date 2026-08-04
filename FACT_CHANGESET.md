# FACT Changeset

This changeset adds a JAX Pi0/Pi0.5 extension for **FACT** in the code. Its three central features are:

1. A configurable logit-normal training-time flow-matching timestep sampler.
2. A current force-torque stream, `ft_current`, that modulates Pi0.5 adaptive RMSNorm.
3. A historical force-torque stream, `ft_history`, whose per-control-step bursts are encoded by a shared temporal convolutional network (TCN) and appended as suffix tokens.

The primary files are:

- `scripts/compute_norm_stats.py`
- `src/openpi/models/gemma.py`
- `src/openpi/models/model.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/policies/fact_policy.py`
- `src/openpi/policies/policy_config.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/weight_loaders.py`
- `src/openpi/transforms.py`

## Detailed file-by-file explanation

### `scripts/compute_norm_stats.py`

The normalization script was refactored so besides the existing `state` and `actions` entries, it can now produce a `force_torque` statistics entry.

The important design decision is that force-torque statistics are **per sensor axis**.

### `src/openpi/models/model.py`

`Observation` now supports two optional fields:

```python
ft_current
ft_history
```

The fields are optional, so existing policies and Pi0 configurations continue to operate without force-torque inputs.

### `src/openpi/models/pi0_config.py`

`Pi0Config` now contains three related groups of options.

#### Force history encoder configuration

```python
ft_window
ft_out_dim
ft_tcn_hidden
ft_history_steps
```

#### Logit-normal scheduler configuration

```python
use_ln_scheduler
ln_alpha
ln_sigma
```

#### Current-force conditioning and contact gating

```python
force_adarms
force_adarms_cg_threshold
force_ft_norm_q01
force_ft_norm_q99
```

The threshold is expressed in raw sensor units. The q01 and q99 statistics are stored in the model configuration so it can invert Pi0.5's quantile normalization before deciding whether contact is present.

### `src/openpi/models/pi0.py`

This file contains most of the new model behavior.

#### Logit-normal sampling

`ln_sample` draws a normal random value, applies the configured scale and shift, maps it through a sigmoid, and clips the result away from exact zero and one:
`compute_loss` selects this sampler only when `use_ln_scheduler=True`; otherwise it retains the original Beta sampler.

Inference integration is unchanged. The scheduler controls which noise levels are observed during training, not the numerical integration schedule used by `sample_actions`.

#### Causal TCN

`CausalDilatedBlock` performs:

```text
left-padded dilated convolution
-> layer normalization
-> swish
-> residual addition
```

The four dilations are:

```text
1, 2, 4, 8
```

With kernel size 3, the receptive field covers:

```text
1 + 2 * (1 + 2 + 4 + 8) = 31 positions
```

#### Shared history encoder

The model constructs a single `ft_tcn_encoder`. It does not construct one encoder per historical step. All steps are folded into the batch dimension, passed through the shared encoder, then restored to the history-token dimension.

#### Suffix attention layout

For FACT's Pi0.5 configuration:

```text
prefix: image and language tokens
suffix block 1: historical FT tokens
suffix block 2: noisy action tokens
```

#### Current-force adaptive RMSNorm path

The current burst is mean-pooled across its within-step samples:

```python
ft_mean = ft_current.reshape(batch, -1, ft_out_dim).mean(axis=1)
```

It then passes through:

```text
Linear(ft_out_dim -> action width)
-> swish
-> Linear(action width -> action width)
-> zero-initialized force_gamma projection
```

The final zero initialization is important so a base Pi0.5 checkpoint retains its original behavior.

#### Contact gating

When `force_adarms_cg_threshold > 0`:

1. The normalized per-axis current-force mean is converted back to raw sensor units.
2. Its L2 magnitude is computed.
3. Samples below the threshold receive:

```python
force_gamma = 0
```

### `src/openpi/models/gemma.py`

Adaptive RMSNorm previously accepted only the timestep embedding. It can now also accept a dictionary:

```python
{
    "time": time_embedding,
    "force_gamma": force_scale_correction,
}
```

The existing timestep projection still produces:

```text
scale, shift, gate
```

Force conditioning modifies only `scale`:

```python
scale = scale + force_gamma
```

### `src/openpi/policies/fact_policy.py`

This new policy module contains:

- `make_fact_example`
- `FACTOutputs`
- `FACTDeltaInputs`
- `FACTInputs`

### `src/openpi/training/data_loader.py`

`DataConfig` can now specify:

```python
history_sequence_keys
history_window
```

The LeRobot loader builds negative timestamps for these keys:

```text
-(history_window - 1) / fps
...
-1 / fps
0
```

FACT uses:

```python
history_sequence_keys=("force_torque",)
history_window=model_config.ft_history_steps + 1
```

### `src/openpi/transforms.py`

FT-specific utilities were added.

#### `alias_ft_norm_stats`

It copies the per-axis `force_torque` statistic object under:

```text
ft_current
ft_history
```

NumPy broadcasting applies the per-axis statistics across all window and history dimensions.

#### `FlattenFTKeys`

This transform runs after normalization. It flattens arrays whose final dimension matches the configured `ft_out_dim`.

### `src/openpi/policies/policy_config.py`

Input and output normalization statistics are separated:

```text
input stats:
  state
  actions
  force_torque
  ft_current alias
  ft_history alias

output stats:
  state
  actions
```

### `src/openpi/training/weight_loaders.py`

`CheckpointWeightLoader` now supports:

```python
extra_init_regex
```

Base Pi0.5 checkpoints do not contain the new FACT parameters. The loader merges matching missing parameters from the freshly initialized model, alongside the existing LoRA behavior.

The FACT configuration initializes:

```text
ft_tcn_encoder
force_adarms_mlp_in
force_adarms_mlp_out
force_gamma_proj
```

These modules are outside the frozen base LLM parameter filter, so they remain trainable.

### `src/openpi/training/config.py`

Two project-specific configurations were added.

#### `pi05_ln_fact_delta_low_mem_finetune`

Adds:

- Logit-normal sampler
- `ln_alpha=1.5`
- `ln_sigma=1.0`

#### `pi05_full_fact_fact_delta_low_mem_finetune`

Adds the full FACT architecture:

- Pi0.5
- Logit-normal scheduler
- Current-force adaptive RMSNorm
- Contact gating at `0.5N`
- Shared within-step TCN
- 27 sensor samples per burst
- Six FT axes
- 30 historical control steps
- One summary token per historical burst

## Compatibility

FACT is JAX-only.
