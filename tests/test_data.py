"""Smoke tests for data transforms."""
import torch

from src.data import get_transforms


def test_train_transform_shape():
    # seed the rng so RandomResizedCrop doesn't flake.
    torch.manual_seed(0)
    tf = get_transforms(image_size=224, train=True)
    from PIL import Image
    img = Image.new("RGB", (320, 320), color=(123, 117, 104))
    out = tf(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0


def test_eval_transform_shape():
    tf = get_transforms(image_size=224, train=False)
    from PIL import Image
    img = Image.new("RGB", (320, 320))
    out = tf(img)
    assert out.shape == (3, 224, 224)
