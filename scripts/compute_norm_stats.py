"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _run_norm_stats(data_config, model_config, batch_size, num_workers, action_horizon, assets_dir, max_frames):
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(data_config, action_horizon, batch_size, max_frames)
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, action_horizon, batch_size, model_config, num_workers, max_frames
        )

    tracked_keys = ["state", "actions"]
    stats: dict[str, normalize.RunningStats] = {}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in tracked_keys:
            if key in batch:
                if key not in stats:
                    stats[key] = normalize.RunningStats()
                stats[key].update(np.asarray(batch[key]))

        ft_stats_source = "ft_current" if "ft_current" in batch else "force_torque"
        if ft_stats_source in batch:
            if "force_torque" not in stats:
                stats["force_torque"] = normalize.RunningStats()
            stats["force_torque"].update(np.asarray(batch[ft_stats_source]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}
    output_path = assets_dir / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)

    data_config = config.data.create(config.assets_dirs, config.model)
    _run_norm_stats(
        data_config, config.model, config.batch_size, config.num_workers,
        config.model.action_horizon, config.assets_dirs, max_frames,
    )


if __name__ == "__main__":
    tyro.cli(main)
