#!/usr/bin/env python3
"""
Smoke tests for the audio SFT pipeline wiring.

These tests validate config resolution, dataset logic, and trainer integration
without requiring GPU, model weights, or real audio data. They prove that the
pipeline components are correctly wired together.

Run with:
    PYTHONPATH=/workspace_yuekai/asr/RL2 python tests/test_audio_sft_smoke.py
"""

import os
import sys
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf, MissingMandatoryValue

# ── Ensure RL2 is on the path ──────────────────────────────────────
RL2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL2_ROOT)


passed = 0
failed = 0


def test(name):
    """Decorator to register and run a test function."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}")
            traceback.print_exc()
            failed += 1
        return fn
    return decorator


# ═══════════════════════════════════════════════════════════════════
# 1. Hydra / OmegaConf Config Resolution
# ═══════════════════════════════════════════════════════════════════
print("\n=== Config Resolution Tests ===")


@test("audio_sft.yaml declares actor.model_name as required (MISSING)")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    # In the raw config, actor.model_name is ??? (MISSING)
    assert OmegaConf.is_missing(audio.actor, "model_name"), \
        "actor.model_name should be MISSING (???) to enforce explicit setting"
    # Base megatron.yaml has model_name: null, but audio_sft.yaml overrides
    # it to ??? via Hydra defaults merge, so users MUST provide it explicitly.
    base = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/actor/megatron.yaml")
    )
    assert base.model_name is None, \
        "Base megatron.yaml should have model_name: null"


@test("actor.model_name: ??? raises MissingMandatoryValue when accessed")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    try:
        _ = audio.actor.model_name
        assert False, "Should have raised MissingMandatoryValue"
    except MissingMandatoryValue:
        pass  # Expected


@test("Config has separate max_length and max_audio_seconds fields")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    assert audio.data.train.max_length == 4096, \
        f"Expected max_length=4096, got {audio.data.train.max_length}"
    assert audio.data.train.max_audio_seconds == 30, \
        f"Expected max_audio_seconds=30, got {audio.data.train.max_audio_seconds}"


@test("Config has all required data fields")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    train = audio.data.train
    required_fields = [
        "dataset_name", "dataset_subset", "dataset_split",
        "batch_size", "max_length", "max_audio_seconds",
        "audio_column", "text_column", "prompt", "remove_text_spaces",
    ]
    for field in required_fields:
        assert hasattr(train, field), f"Missing data.train.{field}"


@test("Config has freeze flags defaulting to false (full SFT)")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    assert audio.actor.freeze_audio_encoder is False
    assert audio.actor.freeze_language_model is False
    assert audio.actor.freeze_multi_modal_projector is False


@test("Config test section inherits from train via interpolation")
def _():
    audio = OmegaConf.load(
        os.path.join(RL2_ROOT, "RL2/trainer/config/audio_sft.yaml")
    )
    # test.max_length is an interpolation of train.max_length
    raw = OmegaConf.to_container(audio, resolve=False)
    assert raw["data"]["test"]["max_length"] == "${data.train.max_length}"
    assert raw["data"]["test"]["max_audio_seconds"] == "${data.train.max_audio_seconds}"


# ═══════════════════════════════════════════════════════════════════
# 2. AudioSFTDataset Logic (no real data needed)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Dataset Logic Tests ===")

from RL2.datasets.audio_sft import (
    _validate_audio, _truncate_audio, build_conversation,
    IGNORE_INDEX, _DEFAULT_MAX_AUDIO_SECONDS, _DEFAULT_SAMPLING_RATE,
)


@test("_validate_audio: dict with array returns (array, sampling_rate)")
def _():
    arr = np.random.randn(16000)
    audio_raw = {"array": arr, "sampling_rate": 16000}
    result_arr, result_sr = _validate_audio(audio_raw, 0)
    assert np.array_equal(result_arr, arr)
    assert result_sr == 16000


@test("_validate_audio: None raises ValueError")
def _():
    try:
        _validate_audio(None, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "None" in str(e)


@test("_validate_audio: empty array raises ValueError")
def _():
    try:
        _validate_audio({"array": np.array([])}, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e)


@test("_validate_audio: dict without 'array' key raises ValueError")
def _():
    try:
        _validate_audio({"waveform": np.array([1.0])}, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "array" in str(e)


@test("_truncate_audio: short audio is unchanged")
def _():
    arr = np.random.randn(1000)
    result = _truncate_audio(arr, 16000, 30)
    assert len(result) == 1000


@test("_truncate_audio: overlong audio is truncated to max_audio_seconds * sr")
def _():
    arr = np.random.randn(600000)  # 37.5 seconds at 16kHz
    result = _truncate_audio(arr, 16000, 30)
    assert len(result) == 480000  # 30 * 16000


@test("_truncate_audio: custom sampling rate is respected")
def _():
    arr = np.random.randn(100000)
    result = _truncate_audio(arr, 8000, 10)
    assert len(result) == 80000  # 10 * 8000


@test("build_conversation: creates correct structure with space removal")
def _():
    conv = build_conversation("Transcribe", "hello world", remove_text_spaces=True)
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"
    assert conv[1]["content"][0]["text"] == "helloworld"


@test("build_conversation: preserves spaces when remove_text_spaces=False")
def _():
    conv = build_conversation("Transcribe", "hello world", remove_text_spaces=False)
    assert conv[1]["content"][0]["text"] == "hello world"


# ═══════════════════════════════════════════════════════════════════
# 3. Boundary-Aware Masking Logic (with synthetic token IDs)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Boundary Masking Tests ===")


@test("_find_assistant_content_boundaries: finds correct span with known token IDs")
def _():
    # Simulate Qwen2 token IDs
    IM_START = 151644
    IM_END = 151645
    # "assistant\n" tokenized as [77091, 198] (approximate; we'll use a mock)
    ASST_HEADER = [77091, 198]

    # Build a synthetic token sequence:
    # [im_start, system_tokens..., im_end, im_start, user_tokens..., im_end,
    #  im_start, assistant_header..., content_tokens..., im_end]
    ids = (
        [IM_START, 100, 101, 102, IM_END]  # system turn
        + [IM_START, 200, 201, 202, IM_END]  # user turn
        + [IM_START] + ASST_HEADER + [300, 301, 302] + [IM_END]  # assistant turn
    )

    # Create a mock dataset to test the method
    class MockDataset:
        def __init__(self):
            self._im_start_id = IM_START
            self._im_end_id = IM_END
            self._assistant_header_ids = ASST_HEADER

    from RL2.datasets.audio_sft import AudioSFTDataset
    # Call the method directly (it's a regular method, not classmethod)
    mock = MockDataset()
    result = AudioSFTDataset._find_assistant_content_boundaries(mock, ids)
    content_start, content_end = result

    # Content should be [300, 301, 302]
    assert content_start == 13, f"Expected content_start=13, got {content_start}"
    assert content_end == 16, f"Expected content_end=16, got {content_end}"
    assert ids[content_start:content_end] == [300, 301, 302]


@test("_find_assistant_content_boundaries: returns (-1,-1) when no im_start found")
def _():
    class MockDataset:
        _im_start_id = 151644
        _im_end_id = 151645
        _assistant_header_ids = [77091, 198]

    from RL2.datasets.audio_sft import AudioSFTDataset
    mock = MockDataset()
    result = AudioSFTDataset._find_assistant_content_boundaries(mock, [1, 2, 3, 4])
    assert result == (-1, -1)


@test("_find_assistant_content_boundaries: returns (-1,-1) when header doesn't match")
def _():
    IM_START = 151644
    IM_END = 151645

    class MockDataset:
        _im_start_id = IM_START
        _im_end_id = IM_END
        _assistant_header_ids = [77091, 198]

    from RL2.datasets.audio_sft import AudioSFTDataset
    mock = MockDataset()
    # Last im_start followed by non-matching header
    ids = [IM_START, 999, 888, IM_END]
    result = AudioSFTDataset._find_assistant_content_boundaries(mock, ids)
    assert result == (-1, -1)


# ═══════════════════════════════════════════════════════════════════
# 4. Trainer Integration (import and filtering)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Trainer Integration Tests ===")


@test("_DIAGNOSTIC_KEYS contains the expected keys")
def _():
    from RL2.trainer.audio_sft import _DIAGNOSTIC_KEYS
    expected = {"input_ids", "attention_mask", "labels", "loss_mask"}
    assert _DIAGNOSTIC_KEYS == expected, \
        f"Expected {expected}, got {_DIAGNOSTIC_KEYS}"


@test("_filter_for_training removes diagnostic keys, keeps training keys")
def _():
    from RL2.trainer.audio_sft import AudioSFTTrainer
    tensor_dict = {
        "states": torch.zeros(2, 10),
        "actions": torch.zeros(2, 10),
        "action_mask": torch.zeros(2, 10),
        "eos_mask": torch.zeros(2, 10),
        "input_ids": torch.zeros(2, 11),
        "attention_mask": torch.zeros(2, 11),
        "labels": torch.zeros(2, 11),
        "loss_mask": torch.zeros(2, 11),
        "input_features": torch.zeros(2, 128, 3000),
        "feature_attention_mask": torch.zeros(2, 3000),
    }
    filtered = AudioSFTTrainer._filter_for_training(tensor_dict)
    # Should keep training + audio keys, remove diagnostic keys
    assert "states" in filtered
    assert "actions" in filtered
    assert "action_mask" in filtered
    assert "eos_mask" in filtered
    assert "input_features" in filtered
    assert "feature_attention_mask" in filtered
    # Diagnostic keys should be removed
    assert "input_ids" not in filtered
    assert "attention_mask" not in filtered
    assert "labels" not in filtered
    assert "loss_mask" not in filtered


@test("AudioSFTDataset module exports are importable")
def _():
    from RL2.datasets.audio_sft import AudioSFTDataset, get_audio_dataloaders
    assert callable(AudioSFTDataset)
    assert callable(get_audio_dataloaders)


@test("datasets __init__ exports AudioSFTDataset and get_audio_dataloaders")
def _():
    from RL2.datasets import AudioSFTDataset, get_audio_dataloaders
    assert callable(AudioSFTDataset)
    assert callable(get_audio_dataloaders)


# ═══════════════════════════════════════════════════════════════════
# 5. Forward Pass Wiring (static analysis)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Forward Pass Wiring Tests ===")


@test("MegatronWorker._build_forward_kwargs passes audio kwargs when present")
def _():
    # Ensure FlashInfer cache exists so Megatron imports succeed
    os.makedirs("/root/.cache/flashinfer/0.5.3/", exist_ok=True)
    from RL2.workers.megatron.base import MegatronWorker
    import torch as _torch
    minibatch = {
        "states": _torch.zeros(1, 5),
        "input_features": _torch.zeros(1, 128, 3000),
        "feature_attention_mask": _torch.zeros(1, 3000),
    }
    kwargs = MegatronWorker._build_forward_kwargs(minibatch, None)
    assert "input_features" in kwargs
    assert "feature_attention_mask" in kwargs
    assert kwargs["input_ids"] is minibatch["states"]
    # Text-only should exclude audio keys
    text_only = {"states": _torch.zeros(1, 5)}
    kwargs2 = MegatronWorker._build_forward_kwargs(text_only, None)
    assert "input_features" not in kwargs2
    assert "feature_attention_mask" not in kwargs2


@test("Freeze flag mapping is correct in MegatronWorker (file content)")
def _():
    source_path = os.path.join(RL2_ROOT, "RL2/workers/megatron/base.py")
    with open(source_path) as f:
        source = f.read()
    assert '"freeze_audio_encoder": "freeze_audio_model"' in source
    assert '"freeze_language_model": "freeze_language_model"' in source
    assert '"freeze_multi_modal_projector": "freeze_audio_projection"' in source


@test("_MULTIMODAL_PASSTHROUGH_KEYS contains audio tensor keys")
def _():
    from RL2.utils.sequences import _MULTIMODAL_PASSTHROUGH_KEYS
    assert "input_features" in _MULTIMODAL_PASSTHROUGH_KEYS
    assert "feature_attention_mask" in _MULTIMODAL_PASSTHROUGH_KEYS


# ═══════════════════════════════════════════════════════════════════
# 6. Executed Collate Tests (AC-1, AC-1.1)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Executed Collate Tests ===")

# Qwen2-Audio special token IDs
_IM_START = 151644
_IM_END = 151645
_AUDIO_TOKEN = 151646
_ASST_HEADER_TOKENS = [77091, 198]


class FakeTokenizer:
    """Minimal tokenizer mock returning deterministic Qwen2-Audio token IDs."""
    _TOKEN_MAP = {
        "<|im_start|>": _IM_START,
        "<|im_end|>": _IM_END,
        "<|AUDIO|>": _AUDIO_TOKEN,
    }

    def convert_tokens_to_ids(self, token):
        return self._TOKEN_MAP.get(token, 0)

    def encode(self, text, add_special_tokens=False):
        if text == "assistant\n":
            return list(_ASST_HEADER_TOKENS)
        return [ord(c) % 1000 for c in text]


class FakeProcessor:
    """Minimal processor mock producing deterministic batch tensors with
    proper chat template structure for boundary detection."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self._last_audios = None

    def apply_chat_template(self, conv, tokenize=False):
        return f"<chat>{conv}</chat>"

    def __call__(self, text=None, audio=None, **kwargs):
        self._last_audios = audio
        batch_size = len(text) if text else len(audio)
        all_ids = []
        for i in range(batch_size):
            ids = (
                [_IM_START, 8948, 198, 2610, 525, _IM_END, 198]
                + [_IM_START, 872, 198, _AUDIO_TOKEN, 1234, 5678, _IM_END, 198]
                + [_IM_START] + list(_ASST_HEADER_TOKENS)
                + [3000 + j for j in range(3 + i)]
                + [_IM_END]
            )
            all_ids.append(ids)
        max_len = max(len(ids) for ids in all_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
        attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]

        mel_lengths = [max(1, len(a) // 160) for a in audio]
        max_mel = max(mel_lengths)
        feats = torch.zeros(batch_size, 128, max_mel)
        feat_mask = torch.zeros(batch_size, max_mel, dtype=torch.long)
        for idx, ml in enumerate(mel_lengths):
            feats[idx, :, :ml] = torch.randn(128, ml)
            feat_mask[idx, :ml] = 1

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "input_features": feats,
            "feature_attention_mask": feat_mask,
        }


def _make_dataset_and_examples(audio_lengths=(16000, 32000)):
    """Create a fake AudioSFTDataset and matching examples."""
    from RL2.datasets.audio_sft import AudioSFTDataset
    proc = FakeProcessor()
    config = OmegaConf.create({
        "audio_column": "audio", "text_column": "text",
        "prompt": "Transcribe", "remove_text_spaces": True,
        "max_length": 4096, "max_audio_seconds": 30,
    })
    fake_hf = type("DS", (), {
        "column_names": ["audio", "text"],
        "__len__": lambda s: 2, "__getitem__": lambda s, i: None,
    })()
    ds = AudioSFTDataset(config, proc, fake_hf)
    examples = [
        {"audio": {"array": np.random.randn(n), "sampling_rate": 16000},
         "text": f"text{i}"}
        for i, n in enumerate(audio_lengths)
    ]
    return ds, examples, proc


@test("collate_fn produces all required keys")
def _():
    ds, examples, _ = _make_dataset_and_examples()
    batch = ds.collate_fn(examples)
    required = {"states", "actions", "action_mask", "eos_mask",
                "input_ids", "attention_mask", "labels", "loss_mask",
                "input_features", "feature_attention_mask"}
    missing = required - set(batch.keys())
    assert not missing, f"Missing keys: {missing}"


@test("collate_fn: input_features has mel dimension 128")
def _():
    ds, examples, _ = _make_dataset_and_examples()
    batch = ds.collate_fn(examples)
    assert batch["input_features"].shape[1] == 128, \
        f"Expected mel dim 128, got {batch['input_features'].shape[1]}"


@test("collate_fn: AUDIO token (151646) appears in input_ids")
def _():
    ds, examples, _ = _make_dataset_and_examples()
    batch = ds.collate_fn(examples)
    for i in range(batch["input_ids"].shape[0]):
        assert _AUDIO_TOKEN in batch["input_ids"][i].tolist(), \
            f"Example {i}: AUDIO token not in input_ids"


@test("collate_fn: labels/loss_mask unmask only assistant content tokens")
def _():
    ds, examples, _ = _make_dataset_and_examples()
    batch = ds.collate_fn(examples)
    loss_mask = batch["loss_mask"]
    labels = batch["labels"]
    # FakeProcessor generates 3+i content tokens per example
    for i, expected_len in enumerate([3, 4]):
        unmasked = (loss_mask[i] == 1.0).sum().item()
        assert unmasked == expected_len, \
            f"Example {i}: expected {expected_len} unmasked, got {unmasked}"
    # Every unmasked label should be a content token (>= 3000)
    for i in range(labels.shape[0]):
        for pos in range(labels.shape[1]):
            if labels[i, pos].item() != IGNORE_INDEX:
                assert labels[i, pos].item() >= 3000, \
                    f"Example {i}, pos {pos}: non-content token unmasked"


@test("collate_fn: feature_attention_mask distinguishes real vs padded frames")
def _():
    ds, examples, _ = _make_dataset_and_examples(audio_lengths=(1600, 32000))
    batch = ds.collate_fn(examples)
    feat_mask = batch["feature_attention_mask"]
    short_real = feat_mask[0].sum().item()
    long_real = feat_mask[1].sum().item()
    assert short_real < long_real, \
        f"Short audio ({short_real} frames) should have fewer real frames than long ({long_real})"
    assert long_real == feat_mask.shape[1], \
        f"Long audio should fill all {feat_mask.shape[1]} frames, got {long_real}"


# ═══════════════════════════════════════════════════════════════════
# 7. Negative Path Dataset Tests (AC-1)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Negative Path Dataset Tests ===")


@test("AudioSFTDataset: missing audio column raises ValueError")
def _():
    from RL2.datasets.audio_sft import AudioSFTDataset
    proc = FakeProcessor()
    config = OmegaConf.create({"audio_column": "audio", "text_column": "text"})
    bad_ds = type("DS", (), {
        "column_names": ["text", "other_col"],
        "__len__": lambda s: 0, "__getitem__": lambda s, i: None,
    })()
    try:
        AudioSFTDataset(config, proc, bad_ds)
        assert False, "Should have raised ValueError for missing audio column"
    except ValueError as e:
        assert "audio" in str(e).lower(), f"Error should mention audio: {e}"


@test("collate_fn: all-invalid audio raises RuntimeError")
def _():
    ds, _, _ = _make_dataset_and_examples()
    invalid = [{"audio": None, "text": "a"}, {"audio": None, "text": "b"}]
    try:
        ds.collate_fn(invalid)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "All examples" in str(e), f"Wrong error: {e}"


@test("collate_fn: overlong audio is truncated without crash")
def _():
    ds, _, proc = _make_dataset_and_examples()
    overlong = [
        {"audio": {"array": np.random.randn(600000), "sampling_rate": 16000}, "text": "a"},
        {"audio": {"array": np.random.randn(700000), "sampling_rate": 16000}, "text": "b"},
    ]
    batch = ds.collate_fn(overlong)
    assert "input_features" in batch, "Should produce valid output"
    for a in proc._last_audios:
        assert len(a) <= 480000, f"Audio not truncated: {len(a)} samples"


# ═══════════════════════════════════════════════════════════════════
# 8. Trainer Wiring Tests (AC-5)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Trainer Wiring Tests ===")


@test("AudioSFTTrainer calls AutoProcessor and sft_step with audio keys")
def _():
    from unittest.mock import MagicMock, patch
    from RL2.trainer.audio_sft import AudioSFTTrainer

    config = OmegaConf.create({
        "actor": {"model_name": "test_model"},
        "data": {},
        "trainer": {
            "load_ckpt_from": None, "save_dir": "/tmp/test_audio_sft",
            "use_wandb": False, "n_epochs": 1, "save_freq": None,
            "project": "test", "experiment_name": "test",
        },
    })

    fake_actor = MagicMock()
    fake_batch = {
        "states": torch.zeros(2, 10), "actions": torch.zeros(2, 10),
        "action_mask": torch.zeros(2, 10), "eos_mask": torch.zeros(2, 10),
        "input_ids": torch.zeros(2, 11), "attention_mask": torch.zeros(2, 11),
        "labels": torch.zeros(2, 11), "loss_mask": torch.zeros(2, 11),
        "input_features": torch.zeros(2, 128, 3000),
        "feature_attention_mask": torch.zeros(2, 3000),
    }

    with patch("RL2.trainer.audio_sft.initialize_actor", return_value=fake_actor), \
         patch("RL2.trainer.audio_sft.AutoProcessor") as MockAP, \
         patch("RL2.trainer.audio_sft.get_audio_dataloaders",
               return_value=([fake_batch], [fake_batch])), \
         patch("RL2.trainer.audio_sft.dist") as md, \
         patch("RL2.trainer.base.dist") as mbd, \
         patch("RL2.trainer.base.wandb"):

        md.get_rank.return_value = 0
        mbd.get_rank.return_value = 0
        MockAP.from_pretrained.return_value = MagicMock()

        trainer = AudioSFTTrainer(config)

        # Verify AutoProcessor called with model_name
        MockAP.from_pretrained.assert_called_once_with(
            "test_model", trust_remote_code=True
        )

        trainer.train()

        # Verify sft_step called with filtered tensor_dict
        assert fake_actor.sft_step.called, "sft_step never called"
        filtered_td = fake_actor.sft_step.call_args_list[0][0][0]
        assert "input_features" in filtered_td, "input_features missing"
        assert "feature_attention_mask" in filtered_td, "feature_attention_mask missing"
        assert "input_ids" not in filtered_td, "Diagnostic key input_ids not removed"
        assert "labels" not in filtered_td, "Diagnostic key labels not removed"


# ═══════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
