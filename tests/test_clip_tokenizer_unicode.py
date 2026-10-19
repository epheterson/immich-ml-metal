"""CLIP text must tokenize the way Immich's ML container tokenizes it.

mlx_clip's tokenizer looked characters up in the vocabulary directly instead
of through CLIP's byte-to-unicode map, so a Smart Search for "мария" (or any
Cyrillic, CJK or emoji text) raised KeyError and returned a 500 (#194).
The container encodes with the model's tokenizer.json, so that is the
reference: same token ids for every script, truncated to 77 with the end
token kept.
"""

from pathlib import Path

import pytest

from src.models.clip import _ByteLevelTokenizer

QUERIES = [
    "мария",
    "Мария на пляже в Сочи",
    "東京の夜景",
    "北京 长城",
    "🐶 dog at the beach 🏖️",
    "café crème brûlée",
    "  Two   spaces\tand a tab  ",
    "a red car",
    "очень длинный запрос " * 20,  # far past 77 tokens
]


def _snapshot(repo: str) -> Path | None:
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
    )
    for snap in sorted(root.glob("*")) if root.exists() else []:
        return snap
    return None


@pytest.fixture(scope="module")
def tokenizers():
    """mlx_clip's tokenizer (ours wraps it) and the reference tokenizer.json."""
    pytest.importorskip("mlx_clip")
    hf = pytest.importorskip("tokenizers")
    from mlx_clip.tokenizer import CLIPTokenizer

    mlx_dir = _snapshot("mlx-community/clip-vit-base-patch32")
    ref_dir = _snapshot("openai/clip-vit-base-patch32")
    if not (
        mlx_dir
        and (mlx_dir / "vocab.json").exists()
        and ref_dir
        and (ref_dir / "tokenizer.json").exists()
    ):
        pytest.skip("CLIP ViT-B/32 tokenizer files are not in the Hugging Face cache")
    ref = hf.Tokenizer.from_file(str(ref_dir / "tokenizer.json"))
    ref.enable_truncation(max_length=_ByteLevelTokenizer.CONTEXT_LENGTH)
    return CLIPTokenizer.from_pretrained(str(mlx_dir)), ref


@pytest.mark.parametrize("text", QUERIES)
def test_token_ids_match_the_containers_tokenizer(tokenizers, text):
    inner, ref = tokenizers
    assert _ByteLevelTokenizer(inner).encode(text) == ref.encode(text).ids


def test_the_unfixed_tokenizer_fails_where_the_fix_does_not(tokenizers):
    """Guards the premise: mlx_clip alone still raises on Cyrillic."""
    inner, _ = tokenizers
    with pytest.raises(KeyError):
        inner.tokenize("мария")
    ids = _ByteLevelTokenizer(inner)(["мария"])
    assert ids.shape[0] == 1 and ids.shape[1] > 2


def test_long_text_is_truncated_with_the_end_token_kept(tokenizers):
    inner, _ = tokenizers
    ids = _ByteLevelTokenizer(inner).encode("очень длинный запрос " * 20)
    assert len(ids) == 77
    assert ids[0] == inner.bos_token and ids[-1] == inner.eos_token
