# Copyright (c) 2025 Resemble AI
# MIT License
"""Regression tests for T3.loss's next-token shift + class-dim fix.

These tests deliberately avoid constructing a real T3 (500M+ params, needs a
LlamaModel/GPT2Model backbone) - they build a bare, un-initialized T3
instance via ``object.__new__`` and monkeypatch ``forward`` to return
hand-crafted logits, then call the REAL (unmodified) ``T3.loss`` on top of
those. This exercises the exact post-forward masking/shift/cross-entropy
logic under test, in well under a second, on CPU only.
"""
import torch
import torch.nn.functional as F

from chatterbox.models.t3.t3 import T3
from chatterbox.models.utils import AttrDict


def _bare_t3(text_logits: torch.Tensor, speech_logits: torch.Tensor) -> T3:
    """A T3 instance with no submodules built - `forward` is stubbed to
    return the given canned logits so `loss()` can be exercised in
    isolation."""
    t3 = object.__new__(T3)  # skip __init__: no transformer/embeddings built
    t3.forward = lambda **kwargs: AttrDict(
        text_logits=text_logits,
        speech_logits=speech_logits,
    )
    return t3


def test_text_loss_targets_next_token_not_same_position():
    """If text_logits[:, j] strongly predicts text_tokens[:, j+1] (the
    correct next-token target) but NOT text_tokens[:, j] (the old, buggy
    same-position target), loss_text must come out near zero. Under the
    original unshifted-target bug this would instead be large (and, before
    the class-dim fix, would raise a shape error outright)."""
    vocab = 20
    text_tokens = torch.tensor([[10, 11, 12, 13]])  # (B=1, len_text=4)
    text_token_lens = torch.tensor([4])

    len_text = text_tokens.size(1)
    text_logits = torch.full((1, len_text, vocab), -10.0)
    # position j should predict text_tokens[j+1] (valid for j=0,1,2; j=3 is
    # the final real position and has no next-token target, so it's masked
    # out of the loss - stuff it with a confidently WRONG prediction to
    # prove the mask, not luck, is what excludes it).
    for j in range(len_text - 1):
        text_logits[0, j, text_tokens[0, j + 1]] = 20.0
    text_logits[0, 3, 5] = 20.0  # confidently wrong / masked position

    speech_tokens = torch.zeros(1, 1, dtype=torch.long)
    speech_token_lens = torch.tensor([1])
    speech_logits = torch.zeros(1, 1, 4)

    t3 = _bare_t3(text_logits, speech_logits)
    loss_text, _ = T3.loss(
        t3,
        t3_cond=None,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
    )
    assert loss_text.item() < 1e-3, f"expected near-zero loss for correctly shifted predictions, got {loss_text.item()}"


def test_text_loss_high_when_logits_only_match_same_position():
    """The mirror-image check: logits that confidently predict the OLD,
    buggy same-position target (text_tokens[:, j] at position j) rather
    than the correct next-token target must produce a large loss under the
    fixed implementation - this is exactly the case the original bug got
    wrong (it would have reported near-zero loss here)."""
    vocab = 20
    text_tokens = torch.tensor([[10, 11, 12, 13]])
    text_token_lens = torch.tensor([4])
    len_text = text_tokens.size(1)

    text_logits = torch.full((1, len_text, vocab), -10.0)
    for j in range(len_text):
        text_logits[0, j, text_tokens[0, j]] = 20.0  # same-position "identity" shortcut

    speech_tokens = torch.zeros(1, 1, dtype=torch.long)
    speech_token_lens = torch.tensor([1])
    speech_logits = torch.zeros(1, 1, 4)

    t3 = _bare_t3(text_logits, speech_logits)
    loss_text, _ = T3.loss(
        t3,
        t3_cond=None,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
    )
    assert loss_text.item() > 5.0, f"expected large loss for same-position identity shortcut, got {loss_text.item()}"


def test_speech_loss_targets_next_token_and_masks_padding():
    """Same shift check for speech_logits/speech_tokens, plus padding-mask
    behavior: a batch row shorter than len_speech should have its padded
    tail excluded from the loss regardless of what garbage logits sit
    there."""
    vocab = 6561
    # row0: fully real, length 3. row1: real length 2, padded to len_speech=3.
    speech_tokens = torch.tensor([
        [1, 2, 3],
        [4, 5, 0],  # position 2 is padding
    ])
    speech_token_lens = torch.tensor([3, 2])
    len_speech = speech_tokens.size(1)

    speech_logits = torch.full((2, len_speech, vocab), -10.0)
    # row0: position 0 -> predict token[1]=2; position1 -> predict token[2]=3;
    # position2 is the final real position (no next-token target), stuff wrong.
    speech_logits[0, 0, 2] = 20.0
    speech_logits[0, 1, 3] = 20.0
    speech_logits[0, 2, 999] = 20.0
    # row1: only position0 has a valid target (token_len=2 -> only j=0 unmasked);
    # position0 -> predict token[1]=5. positions 1,2 masked - stuff wrong.
    speech_logits[1, 0, 5] = 20.0
    speech_logits[1, 1, 999] = 20.0
    speech_logits[1, 2, 999] = 20.0

    text_tokens = torch.tensor([[10, 13], [10, 13]])
    text_token_lens = torch.tensor([2, 2])
    text_logits = torch.zeros(2, 2, 20)

    t3 = _bare_t3(text_logits, speech_logits)
    _, loss_speech = T3.loss(
        t3,
        t3_cond=None,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
    )
    assert loss_speech.item() < 1e-3, f"expected near-zero loss (masked padding excluded), got {loss_speech.item()}"


def test_class_dim_is_at_position_one_for_cross_entropy():
    """Direct shape check: `T3.loss` must feed F.cross_entropy a
    (B, vocab, len) tensor, not the raw (B, len, vocab) `forward()` output -
    verified by using distinct len/vocab sizes so a wrong ordering would
    either raise (mismatched target shape) or silently compute over the
    wrong axis. Reproduces the manual F.cross_entropy call to confirm exact
    numerical agreement with `loss()`'s internal computation."""
    B, len_text, vocab = 1, 4, 37  # deliberately len_text != vocab
    text_tokens = torch.tensor([[1, 2, 3, 4]])
    text_token_lens = torch.tensor([len_text])
    torch.manual_seed(0)
    text_logits = torch.randn(B, len_text, vocab)

    speech_tokens = torch.zeros(1, 1, dtype=torch.long)
    speech_token_lens = torch.tensor([1])
    speech_logits = torch.zeros(1, 1, 4)

    t3 = _bare_t3(text_logits, speech_logits)
    loss_text, _ = T3.loss(
        t3,
        t3_cond=None,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
    )

    IGNORE_ID = -100
    expected_targets = F.pad(text_tokens[:, 1:], (0, 1), value=IGNORE_ID)
    mask = torch.arange(len_text)[None] >= (text_token_lens[:, None] - 1)
    expected_targets = expected_targets.masked_fill(mask, IGNORE_ID)
    expected = F.cross_entropy(text_logits.transpose(1, 2), expected_targets, ignore_index=IGNORE_ID)

    assert torch.allclose(loss_text, expected), (loss_text.item(), expected.item())
