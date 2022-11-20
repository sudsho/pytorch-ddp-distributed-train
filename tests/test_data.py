"""Smoke tests for data transforms."""
import torch

from src.data import get_transforms


def test_train_transform_shape():
    tf = get_transforms(image_size=224, train=True)
    # mock a PIL-ish image via tensor->PIL roundtrip is overkill, just call
    # the underlying ops on a dummy uint8 tensor through ToPILImage at use time.
    # Here we just check the pipeline exists and is callable.
    from PIL import Image
    img = Image.new("RGB", (320, 320), color=(123, 117, 104))
    out = tf(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_eval_transform_shape():
    tf = get_transforms(image_size=224, train=False)
    from PIL import Image
    img = Image.new("RGB", (320, 320))
    out = tf(img)
    assert out.shape == (3, 224, 224)
