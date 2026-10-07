from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.hunyuan import xFuserHunyuanvideo15Model
from xfuser.model_executor.models.runner_models.z_image import (
    xFuserZImageModel,
    xFuserZImageTurboModel,
)


class _RecordingPipe:
    def __init__(self, guider=None):
        self.guider = guider
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=["image"], frames=["video"])


def _runner(model_cls, pipe, task="t2v"):
    model = object.__new__(model_cls)
    model.config = SimpleNamespace(task=task, batch_size=None)
    model.pipe = pipe
    model._make_generator = lambda seed: None
    return model


@pytest.mark.parametrize("model_cls", [xFuserZImageModel, xFuserZImageTurboModel])
def test_z_image_forwards_negative_prompt(model_cls):
    pipe = _RecordingPipe()
    input_args = {
        "height": 64,
        "width": 64,
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 0,
    }

    _runner(model_cls, pipe)._run_pipe(input_args)

    assert pipe.calls[0]["negative_prompt"] == "blurry"


def _hv15_args(**overrides):
    args = {
        "num_inference_steps": 2,
        "num_frames": 5,
        "seed": 0,
        "prompt": "a cat",
        "height": 64,
        "width": 64,
    }
    args.update(overrides)
    return args


def test_hunyuanvideo15_forwards_negative_prompt_and_guidance_scale():
    from diffusers.guiders import ClassifierFreeGuidance

    pipe = _RecordingPipe(guider=ClassifierFreeGuidance(guidance_scale=6.0))

    _runner(xFuserHunyuanvideo15Model, pipe)._run_pipe(_hv15_args(negative_prompt="blurry", guidance_scale=2.5))

    assert pipe.calls[0]["negative_prompt"] == "blurry"
    assert pipe.guider.guidance_scale == 2.5


def test_hunyuanvideo15_keeps_checkpoint_guidance_without_flag():
    from diffusers.guiders import ClassifierFreeGuidance

    guider = ClassifierFreeGuidance(guidance_scale=6.0)
    pipe = _RecordingPipe(guider=guider)

    _runner(xFuserHunyuanvideo15Model, pipe)._run_pipe(_hv15_args())

    assert pipe.guider is guider
    assert pipe.calls[0]["negative_prompt"] is None
