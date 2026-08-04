# FACT training and inference

FACT is currently JAX-only and requires an NVIDIA GPU for post-training.

## 1. Prepare the LeRobot dataset

The full FACT config expects these per-frame features:

| Feature | Shape | Meaning |
| --- | --- | --- |
| `left_image` | `(H, W, 3)` | External RGB image, preferably `uint8` |
| `wrist_image` | `(H, W, 3)` | Wrist RGB image, preferably `uint8` |
| `state` | `(17,)` | Joint position `(7)`, end-effector position `(3)`, rotation-6D `(6)`, gripper `(1)` |
| `force_torque` | `(27, 6)` | One control step of force `(3)` and torque `(3)` samples |
| `actions` | `(10,)` | Target position `(3)`, rotation-6D `(6)`, gripper `(1)` |
| `task` | string | Language instruction materialized by LeRobot from `task_index` and `meta/tasks.jsonl` |

Store one `(27, 6)` force-torque burst per dataset frame. The data loader constructs the 31-step sequence (30 previous bursts plus the current burst) during sampling and clamps history at episode boundaries. The dataset FPS must match the control FPS used when collecting the trajectories.

Before training on a different dataset, update `repo_id`, `asset_id`, and `assets_dir` in the relevant FACT config in [`src/openpi/training/config.py`](../../src/openpi/training/config.py). `assets_dir` should be an absolute path to this repository's `assets/<config-name>` directory, and `asset_id` should normally equal `repo_id`.

The available configs are:

| Config | Force input | Timestep sampler |
| --- | --- | --- |
| `pi05_ln_fact_delta_low_mem_finetune` | No | Logit-normal |
| `pi05_full_fact_fact_delta_low_mem_finetune` | Current FT plus 30-step FT history | Logit-normal |

Use `pi05_full_fact_fact_delta_low_mem_finetune` for the complete FACT model.

## 3. Compute normalization statistics

Compute statistics with the same config that will be used for training:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_full_fact_fact_delta_low_mem_finetune
```

## 4. Train

Start a new run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_full_fact_fact_delta_low_mem_finetune \
  --exp-name=<experimenta_name> \
  --overwrite
```

## 5. Run Inference

Follow the OpenPI examples to write a policy client that adapts to your sensors and robot, formats observations as specified above, and sends them to the policy server.