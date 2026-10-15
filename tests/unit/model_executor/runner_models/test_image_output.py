"""Image runners take tensors from the pipeline and convert them to PIL on the device.

output_type="pil" makes diffusers scale, round and cast the image in NumPy on the host,
inside the timed pipe. The runners ask for output_type="pt" and use images_to_pil instead,
which must give the same pixels.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image

from xfuser.model_executor.models.runner_models.base_model import images_to_pil
from xfuser.model_executor.models.runner_models.flux import xFuserFlux2Klein9BModel
from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel, xFuserZImageTurboModel


def _decoded(batch=2, height=24, width=40):
    # bf16 VAE output: [-1, 1] plus some overshoot that postprocess clamps
    torch.manual_seed(0)
    return (torch.rand(batch, 3, height, width) * 2.2 - 1.1).to(torch.bfloat16)


def test_images_to_pil_matches_diffusers_pil_postprocess():
    processor = VaeImageProcessor()
    decoded = _decoded()

    expected = processor.postprocess(decoded, output_type="pil")
    images = images_to_pil(processor.postprocess(decoded, output_type="pt"))

    assert all(isinstance(image, Image) for image in images)
    assert [image.size for image in images] == [image.size for image in expected]
    for image, ref in zip(images, expected):
        assert image.mode == ref.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), np.asarray(ref))


def test_images_to_pil_rounds_midpoints_like_numpy():
    # x * 255 at or next to .5: torch must round each value the way NumPy does
    levels = torch.tensor([0.5, 1.5, 2.5, 127.5, 128.5, 253.5, 254.5, 255.0]) / 255
    images = levels.view(1, 1, 1, -1).expand(1, 3, 2, -1).contiguous()

    expected = VaeImageProcessor.numpy_to_pil(images.permute(0, 2, 3, 1).numpy())

    np.testing.assert_array_equal(np.asarray(images_to_pil(images)[0]), np.asarray(expected[0]))


def test_images_to_pil_leaves_non_tensors_alone():
    images = ["already converted"]

    assert images_to_pil(images) is images


class _TensorPipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=torch.rand(1, 3, 16, 32))


@pytest.mark.parametrize("model_cls", [xFuserZImageModel, xFuserZImageTurboModel, xFuserFlux2Klein9BModel])
def test_image_runners_request_tensors_and_return_pil(model_cls):
    model = object.__new__(model_cls)
    model.config = SimpleNamespace(batch_size=None)
    model.pipe = _TensorPipe()
    model._make_generator = lambda seed: None
    input_args = {
        "height": 16,
        "width": 32,
        "prompt": "a cat",
        "images": None,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "seed": 0,
    }

    output = model._run_pipe(input_args)

    assert model.pipe.calls[0]["output_type"] == "pt"
    assert len(output.images) == 1
    assert isinstance(output.images[0], Image)
    assert output.images[0].size == (32, 16)
