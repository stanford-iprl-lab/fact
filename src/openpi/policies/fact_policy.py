import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_fact_example() -> dict:
    """Creates a random input example for the FACT policy."""
    return {
        "observation/left_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(17),
        "observation/ft_sequence": np.random.rand(31, 27, 6).astype(np.float32),
        "actions": np.random.rand(10),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FACTOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}


@dataclasses.dataclass(frozen=True)
class FACTDeltaInputs(transforms.DataTransformFn):

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/left_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        raw_state = np.asarray(data["observation/state"])
        # Reorder: EE state first, then q joints
        state = np.concatenate([raw_state[7:], raw_state[:7]])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FACTInputs(transforms.DataTransformFn):
    """Input transform for the FACT architecture.
    """

    model_type: _model.ModelType
    ft_history_steps: int  # number of past control steps for suffix tokens
    ft_window: int = 27           # FT readings per control step
    ft_out_dim: int = 6

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/left_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        raw_state = np.asarray(data["observation/state"])
        state = np.concatenate([raw_state[7:], raw_state[:7]])  # (17,)

        # ft_sequence: (ft_history_steps + 1, ft_window, ft_out_dim)
        ft_sequence = np.asarray(data["observation/ft_sequence"])
        total = self.ft_history_steps + 1

        # Current burst
        ft_current = ft_sequence[-1]  # (ft_window, ft_out_dim)

        # Past ft_history_steps bursts, stacked along the window dim
        ft_past = ft_sequence[-total:-1]  # (ft_history_steps, ft_window, ft_out_dim)
        ft_history = ft_past.reshape(-1, self.ft_out_dim)  # (ft_history_steps * ft_window, ft_out_dim)

        inputs = {
            "state": state,
            "ft_current": ft_current,
            "ft_history": ft_history,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
