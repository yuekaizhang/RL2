#!/usr/bin/env python3
"""
AC-2 Runtime Tests: MegatronWorker forward pass with Qwen2-Audio.

These tests load the actual Qwen2-Audio model on a single GPU and validate:
1. Forward pass accepts audio tensors (input_features, feature_attention_mask)
2. Text-only forward pass remains backward compatible
3. Malformed input_features shape produces a clear error
4. Freeze flags correctly prevent gradient flow to frozen components
5. _build_forward_kwargs helper correctly constructs forward arguments

Prerequisites:
    - GPU available (CUDA)
    - Model weights at /workspace_yuekai/HF/Qwen2-Audio-7B
    - Megatron-Bridge and RL2 on PYTHONPATH

Run with:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace_yuekai/asr/RL2:/workspace_yuekai/asr/Megatron-Bridge \
    /workspace_yuekai/asr/Megatron-Bridge/.venv/bin/python tests/test_ac2_runtime.py
"""

import os
import sys
import traceback

# Set distributed env before any torch imports
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29507")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

RL2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL2_ROOT)

HF_MODEL = "/workspace_yuekai/HF/Qwen2-Audio-7B"
AUDIO_TOKEN_ID = 151646

passed = 0
failed = 0


def test(name):
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
# 0. Prerequisite checks
# ═══════════════════════════════════════════════════════════════════
print("\n=== Prerequisite Checks ===")

if not torch.cuda.is_available():
    print("  SKIP: No GPU available. AC-2 runtime tests require CUDA.")
    print(f"\n{'='*60}")
    print(f"Results: 0 passed, 0 failed, 0 total (SKIPPED - no GPU)")
    print(f"{'='*60}")
    sys.exit(0)

if not os.path.isdir(HF_MODEL):
    print(f"  SKIP: Model not found at {HF_MODEL}")
    print(f"\n{'='*60}")
    print(f"Results: 0 passed, 0 failed, 0 total (SKIPPED - no model)")
    print(f"{'='*60}")
    sys.exit(0)

print("  GPU available, model weights found. Proceeding.")


# ═══════════════════════════════════════════════════════════════════
# 1. _build_forward_kwargs Tests (lightweight, no model needed)
# ═══════════════════════════════════════════════════════════════════
print("\n=== _build_forward_kwargs Tests ===")

# Ensure FlashInfer cache exists
os.makedirs("/root/.cache/flashinfer/0.5.3/", exist_ok=True)

from RL2.workers.megatron.base import MegatronWorker


@test("_build_forward_kwargs includes audio keys when present in minibatch")
def _():
    minibatch = {
        "states": torch.zeros(2, 10),
        "actions": torch.zeros(2, 10),
        "input_features": torch.zeros(2, 128, 3000),
        "feature_attention_mask": torch.zeros(2, 3000),
    }
    kwargs = MegatronWorker._build_forward_kwargs(minibatch, None)
    assert "input_features" in kwargs, "input_features not in forward_kwargs"
    assert "feature_attention_mask" in kwargs
    assert kwargs["input_ids"] is minibatch["states"]
    assert kwargs["input_features"] is minibatch["input_features"]


@test("_build_forward_kwargs excludes audio keys for text-only minibatch")
def _():
    minibatch = {"states": torch.zeros(2, 10), "actions": torch.zeros(2, 10)}
    kwargs = MegatronWorker._build_forward_kwargs(minibatch, None)
    assert "input_features" not in kwargs
    assert "feature_attention_mask" not in kwargs
    assert "input_ids" in kwargs


@test("_build_forward_kwargs passes packed_seq_params through")
def _():
    minibatch = {"states": torch.zeros(1, 5)}
    sentinel = object()
    kwargs = MegatronWorker._build_forward_kwargs(minibatch, sentinel)
    assert kwargs["packed_seq_params"] is sentinel


# ═══════════════════════════════════════════════════════════════════
# 2. Model Loading and Forward Pass Tests (requires GPU + model)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Model Forward Pass Tests ===")

# Initialize distributed
torch.distributed.init_process_group("nccl", rank=0, world_size=1)

from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig

# Load model once for all tests
print("  Loading Qwen2-Audio model...")
bridge = AutoBridge.from_hf_pretrained(HF_MODEL)
provider = bridge.to_megatron_provider()
provider.params_dtype = torch.bfloat16
provider.autocast_dtype = torch.bfloat16
provider.pipeline_dtype = torch.bfloat16
provider.bf16 = True
provider.attention_backend = "flash"
provider.variable_seq_lengths = True
provider.moe_token_dispatcher_type = "alltoall"
# Freeze audio encoder for testing
provider.freeze_audio_model = True
provider.freeze_language_model = False
provider.freeze_audio_projection = False
provider.finalize()
provider.initialize_model_parallel(seed=42)

ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=True, overlap_grad_reduce=False,
    use_distributed_optimizer=False)
models = provider.provide_distributed_model(
    ddp_config=ddp_config, wrap_with_ddp=False)
model = models[0]
inner = model.module if hasattr(model, "module") else model
print(f"  Model loaded: {type(inner).__name__}")


def _audio_output_length(mel_len):
    """Compute audio output token count using Qwen2-Audio's downsampling formula."""
    feat_len = (mel_len - 1) // 2 + 1
    return (feat_len - 2) // 2 + 1


MEL_LEN = 3000  # Whisper encoder requires exactly 3000 mel frames
N_AUDIO_TOKENS = _audio_output_length(MEL_LEN)  # 750


def _make_audio_input_ids(batch_size=1, n_audio_tokens=N_AUDIO_TOKENS, n_text_tokens=20):
    """Create input_ids with consecutive AUDIO tokens (modern processing path)."""
    ids = []
    for _ in range(batch_size):
        # System tokens + user tokens + audio tokens + assistant tokens
        seq = (
            [151644, 8948, 198, 2610, 525, 151645, 198]  # system
            + [151644, 872, 198]  # user start
            + [AUDIO_TOKEN_ID] * n_audio_tokens  # consecutive audio tokens
            + [1234, 5678, 151645, 198]  # user end
            + [151644, 77091, 198]  # assistant start
            + list(range(3000, 3000 + n_text_tokens))  # content
            + [151645]  # assistant end
        )
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long).cuda()


@test("Forward pass with audio tensors produces 3D output")
def _():
    input_ids = _make_audio_input_ids(batch_size=1)
    feats = torch.randn(1, 128, MEL_LEN, dtype=torch.bfloat16).cuda()
    feat_mask = torch.ones(1, MEL_LEN, dtype=torch.long).cuda()
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=None,
            position_ids=None, labels=None,
            input_features=feats, feature_attention_mask=feat_mask)
    assert out.dim() == 3, f"Expected 3D output, got {out.dim()}D shape={out.shape}"
    # Output is [batch, seq_len, vocab_size]
    assert out.shape[0] == 1, f"Batch dim should be 1, got {out.shape[0]}"
    assert out.shape[2] > 100000, f"Vocab dim too small: {out.shape[2]}"


@test("Text-only forward pass (backward compatibility)")
def _():
    input_ids = torch.randint(100, 1000, (1, 20)).cuda()
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=None,
            position_ids=None, labels=None)
    assert out.dim() == 3, f"Expected 3D output, got {out.dim()}D shape={out.shape}"
    # Output is [batch, seq_len, vocab_size]
    assert out.shape[0] == 1, f"Batch dim should be 1, got {out.shape[0]}"
    assert out.shape[1] == 20, f"Seq dim should be 20, got {out.shape[1]}"


@test("Malformed input_features shape causes runtime error")
def _():
    input_ids = _make_audio_input_ids(batch_size=1)
    # Wrong mel dimension (64 instead of expected 128) - Conv1d expects 128 channels
    bad_feats = torch.randn(1, 64, MEL_LEN, dtype=torch.bfloat16).cuda()
    feat_mask = torch.ones(1, MEL_LEN, dtype=torch.long).cuda()
    raised = False
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids, attention_mask=None,
                position_ids=None, labels=None,
                input_features=bad_feats, feature_attention_mask=feat_mask)
    except (RuntimeError, ValueError):
        raised = True
    assert raised, (
        "Model unexpectedly accepted input_features with wrong mel dimension "
        "(64 instead of 128). Malformed shapes must cause an error."
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Freeze and Gradient Tests
# ═══════════════════════════════════════════════════════════════════
print("\n=== Freeze and Gradient Tests ===")


@test("Frozen audio encoder has requires_grad=False on all parameters")
def _():
    total = sum(1 for p in inner.audio_tower.parameters())
    frozen = sum(1 for p in inner.audio_tower.parameters() if not p.requires_grad)
    assert frozen == total, f"Expected all {total} frozen, got {frozen}"


@test("Unfrozen language model has requires_grad=True on all parameters")
def _():
    total = sum(1 for p in inner.language_model.parameters())
    trainable = sum(1 for p in inner.language_model.parameters() if p.requires_grad)
    assert trainable == total, f"Expected all {total} trainable, got {trainable}"


@test("Gradient flows to LM but not to frozen audio encoder")
def _():
    model.zero_grad()
    input_ids = _make_audio_input_ids(batch_size=1)
    feats = torch.randn(1, 128, MEL_LEN, dtype=torch.bfloat16).cuda()
    feat_mask = torch.ones(1, MEL_LEN, dtype=torch.long).cuda()
    out = model(
        input_ids=input_ids, attention_mask=None,
        position_ids=None, labels=None,
        input_features=feats, feature_attention_mask=feat_mask)
    loss = out.sum()
    loss.backward()
    # Audio encoder should have no gradients (frozen)
    audio_grads = sum(
        1 for p in inner.audio_tower.parameters() if p.grad is not None)
    assert audio_grads == 0, f"Frozen audio has {audio_grads} params with grad"
    # LM should have gradients
    lm_grads = sum(
        1 for p in inner.language_model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0)
    assert lm_grads > 0, f"LM has no params with non-zero gradient"


# ═══════════════════════════════════════════════════════════════════
# Cleanup and Results
# ═══════════════════════════════════════════════════════════════════
torch.distributed.destroy_process_group()

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
