"""The fallback engine must shape images the way Immich's ML container does.

The container resizes the short side and center-crops for every CLIP model,
ignoring the resize_mode in the model's preprocessing config. open_clip honours
it, and for SigLIP/SigLIP2 webli it says "squash". A squashed portrait embeds
as a different picture, so these tests build open_clip-shaped pipelines and
check that what comes out is a crop, not a stretch.
"""

import torch
from PIL import Image
from torchvision import transforms as T

from src.models.clip import _preprocess_like_immich


def _banded_portrait():
    """16x32 portrait: green top quarter, red middle half, green bottom quarter.

    A center crop keeps only the red. A squash keeps the green bands too.
    """
    img = Image.new("RGB", (16, 32), (0, 255, 0))
    img.paste((255, 0, 0), (0, 8, 16, 24))
    return img


def test_a_squashing_pipeline_is_made_to_crop():
    """The SigLIP case: open_clip builds Resize((n, n)), which stretches."""
    squash = T.Compose([T.Resize((8, 8)), T.ToTensor()])
    assert squash(_banded_portrait())[1].max() > 0.9, "control: squash keeps green"

    out = _preprocess_like_immich(squash)(_banded_portrait())
    assert out.shape == (3, 8, 8)
    assert out[1].max() < 0.1, "green survived, so the portrait was squashed"
    assert out[0].min() > 0.9, "the kept region should be the red center"


def test_the_crop_keeps_the_aspect_ratio_rather_than_stretching():
    """A landscape image with a centered square must come back as that square."""
    img = Image.new("RGB", (64, 32), (0, 0, 255))
    img.paste((255, 0, 0), (16, 0, 48, 32))  # centered 32x32 red square
    out = _preprocess_like_immich(T.Compose([T.Resize((8, 8)), T.ToTensor()]))(img)
    assert out[2].max() < 0.1, "blue side margins leaked in: it stretched"


def test_normalisation_from_the_model_config_is_kept():
    """Only the geometry changes. mean and std still come from the weights.

    Compared exactly against the same crop without normalisation, since
    bicubic ringing at the band edges makes absolute pixel checks brittle.
    """
    mean, std = (0.5, 0.4, 0.3), (0.5, 0.25, 0.2)
    plain = _preprocess_like_immich(T.Compose([T.Resize((8, 8)), T.ToTensor()]))
    normed = _preprocess_like_immich(
        T.Compose([T.Resize((8, 8)), T.ToTensor(), T.Normalize(mean, std)])
    )
    img = _banded_portrait()
    expected = (plain(img) - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    assert torch.allclose(normed(img), expected, atol=1e-6)


def test_a_resize_then_crop_pipeline_is_also_normalised_to_immichs_geometry():
    """OpenAI-style weights already crop, via torchvision's own rounding.

    They go through the same path so every model matches the container the
    same way, rather than two slightly different crops.
    """
    pipe = T.Compose([T.Resize(8), T.CenterCrop(8), T.ToTensor()])
    out = _preprocess_like_immich(pipe)(_banded_portrait())
    assert out.shape == (3, 8, 8)
    assert out[1].max() < 0.1


def test_a_pipeline_with_no_leading_geometry_is_left_alone():
    pipe = T.Compose([T.ToTensor()])
    assert _preprocess_like_immich(pipe) is pipe


def test_the_fallback_loader_actually_applies_it():
    """Testing the helper proves nothing if the loader stops calling it.

    Drive the real _load_fallback with open_clip returning a SigLIP-style
    squashing pipeline, and check the processor it installs crops.
    """
    from unittest.mock import MagicMock, patch
    import threading

    from src.models import clip

    c = clip.MLXClip.__new__(clip.MLXClip)
    c.model_name = "ViT-B-16-SigLIP__webli"
    c._inference_lock = threading.Lock()
    squash = T.Compose([T.Resize((8, 8)), T.ToTensor()])
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    with patch("open_clip.create_model_and_transforms", return_value=(fake_model, None, squash)), \
            patch("open_clip.get_tokenizer", return_value=MagicMock()):
        c._load_fallback()
    try:
        out = c._processor(_banded_portrait())
        assert out[1].max() < 0.1, "the loader installed open_clip's squash unchanged"
    finally:
        c._accumulator.stop()
