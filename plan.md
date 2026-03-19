# Integrate Qwen2-Audio SFT Training into RL2 Framework

## Goal Description

Integrate Megatron-based audio LLM (Qwen2-Audio) supervised fine-tuning into the RL2 framework. This involves:
1. Creating a new `AudioSFTDataset` in RL2 that handles audio data loading, mel spectrogram extraction, and tokenization (following RL2's dataset patterns)
2. Extending `MegatronWorker._forward_step()` to pass audio-specific tensors (`input_features`, `feature_attention_mask`) to the model
3. Creating a YAML configuration file for managing audio SFT training parameters (model, data, freeze strategy, training hyperparameters)
4. Creating a `qwen_audio_sft.sh` shell script that orchestrates the full pipeline: HF-to-Megatron checkpoint conversion, training, and Megatron-to-HF export
5. Training operates with CP=1 (no context parallelism considerations needed)

The implementation leverages the existing Megatron-Bridge `Qwen2AudioModel`, `Qwen2AudioBridge`, and `Qwen2AudioModelProvider` which already support Qwen2-Audio architecture. The RL2 side needs new dataset handling, extended forward pass, and configuration/scripting.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: AudioSFTDataset correctly loads and processes audio data
  - Positive Tests (expected to PASS):
    - Loading `yuekai/aishell` dataset produces samples with `input_ids`, `attention_mask`, `input_features`, `feature_attention_mask`, `labels`, and `loss_mask` tensors
    - `input_features` shape is `[batch, n_mels, mel_seq_len]` (128 mel bins for Qwen2-Audio)
    - `labels` are masked with `IGNORE_INDEX=-100` for all non-assistant tokens (system prompt, user message, audio placeholder tokens)
    - Loss is computed only on assistant response tokens (transcription text)
    - Audio `<|AUDIO|>` token (ID 151646) appears in `input_ids` at the correct positions
    - Chinese text has spaces stripped (aishell ASR format)
  - Negative Tests (expected to FAIL):
    - Attempting to load a dataset without an `audio` column raises an informative error
    - Samples with corrupted/missing audio are handled gracefully (skipped or error reported)
  - AC-1.1: Collation handles variable-length audio and text within a batch
    - Positive: Batches with different audio durations are padded correctly; `feature_attention_mask` correctly reflects real vs. padded positions
    - Negative: Batches where all audio exceeds the processor's max length (30s/480000 samples) are truncated, not crashing

- AC-2: MegatronWorker forward pass supports audio inputs
  - Positive Tests (expected to PASS):
    - When `tensor_dict` contains `input_features` and `feature_attention_mask`, these are passed to `model()` as keyword arguments
    - When `tensor_dict` does NOT contain `input_features` (text-only SFT), the forward pass works identically to before (backward compatible)
    - The `Qwen2AudioModel.forward()` receives `input_features` and `feature_attention_mask` and produces correct output shape `[seq_len, batch, vocab_size]`
  - Negative Tests (expected to FAIL):
    - Passing `input_features` with wrong shape (e.g., missing mel dimension) causes a clear error from the audio encoder
  - AC-2.1: Loss computation is correct for audio SFT
    - Positive: SFT loss is negative log-likelihood computed only over unmasked (assistant response) tokens; gradient flows through audio encoder, projector, and language model when none are frozen
    - Negative: Frozen components have `requires_grad=False` and receive no gradient updates

- AC-3: YAML configuration file manages all training parameters
  - Positive Tests (expected to PASS):
    - A YAML config file exists at `RL2/RL2/trainer/config/audio_sft.yaml` (or similar path within the Hydra config structure)
    - Config includes sections for: model (`model_name`, freeze flags), data (dataset path/subset/split, prompt, audio/text columns), training (batch_size, learning rate, epochs), and optimizer/scheduler settings
    - All config values can be overridden via Hydra CLI dot-notation (e.g., `data.train.dataset_name=my_dataset`)
    - Default freeze strategy is full SFT (all components unfrozen), with `freeze_audio_encoder`, `freeze_language_model`, `freeze_multi_modal_projector` flags available
  - Negative Tests (expected to FAIL):
    - Running without specifying `actor.model_name` produces a clear Hydra/OmegaConf error about missing required config

- AC-4: Shell script `qwen_audio_sft.sh` orchestrates the full training pipeline
  - Positive Tests (expected to PASS):
    - Script sets `PYTHONPATH` to include RL2 and uses `python_path` pointing to Megatron-Bridge venv
    - Script performs HF-to-Megatron checkpoint conversion before training (using `convert_checkpoints.py import`) if Megatron checkpoint directory doesn't already exist
    - Script launches distributed training via `torchrun` (or `torch.distributed.run`) with configurable `nproc_per_node`
    - Script performs Megatron-to-HF export after training completes (using `export_hf.py` or `convert_checkpoints.py export`)
    - Script accepts configurable parameters: HF model path, dataset, number of GPUs, training hyperparameters
  - Negative Tests (expected to FAIL):
    - Running the script without the Megatron-Bridge venv Python available fails with a clear "python not found" error
    - Running with an invalid HF model path fails at the conversion step with a clear error

- AC-5: Audio SFT trainer entry point integrates with RL2 training loop
  - Positive Tests (expected to PASS):
    - A new trainer (e.g., `AudioSFTTrainer`) or extended `SFTTrainer` correctly initializes with `AutoProcessor` (in addition to tokenizer) for audio feature extraction
    - Training loop iterates over `AudioSFTDataset`, calls `actor.sft_step()` with audio tensor_dict, and produces decreasing loss
    - Checkpointing (save/load) works correctly, preserving audio encoder, projector, and language model weights
    - Final model save outputs HF-compatible format
  - Negative Tests (expected to FAIL):
    - Using a text-only model (e.g., `Qwen2.5-0.5B`) with the audio SFT trainer fails with an informative error about missing audio components

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes:
- A fully RL2-native `AudioSFTDataset` class that handles audio loading, mel spectrogram extraction via `Qwen2AudioProcessor`, conversation formatting, label masking, and batch collation
- Generic extension of `MegatronWorker._forward_step()` that passes any extra keys from `tensor_dict` (beyond `states`, `actions`, `action_mask`) to the model, making it reusable for future multimodal models
- A complete Hydra YAML configuration with all tunable parameters: model selection, freeze flags per component, dataset configuration (path, subset, split, prompt, column names), training hyperparameters (lr, batch_size, epochs, warmup), optimizer/scheduler settings
- A `qwen_audio_sft.sh` script with automatic HF-to-Megatron conversion, distributed training launch, and Megatron-to-HF export
- An `AudioSFTTrainer` (or modified `SFTTrainer`) that initializes both tokenizer and processor
- Proper handling of data distribution via RL2's `scatter_data` pattern, adapted for audio tensors

### Lower Bound (Minimum Acceptable Scope)

The implementation includes:
- A working `AudioSFTDataset` that loads audio data and produces the required tensors for `Qwen2AudioModel`
- Minimal modification to `MegatronWorker._forward_step()` to pass `input_features` and `feature_attention_mask` when present
- A YAML config file with essential parameters (model path, dataset, basic training settings, freeze flags)
- A working `qwen_audio_sft.sh` script that runs the full pipeline (convert, train, export)
- The training produces a valid fine-tuned model that can be exported to HF format

### Allowed Choices

- Can use: `Qwen2AudioProcessor` from HuggingFace transformers for audio feature extraction; Megatron-Bridge's `AutoBridge` for checkpoint conversion; RL2's existing Hydra configuration system; RL2's existing `MegatronActor.sft_step()` pattern
- Can use: The `qwen2_audio_collate_fn` pattern from Megatron-Bridge as reference for building RL2's own collation logic
- Cannot use: Megatron-Bridge's `HFDatasetConversationProvider` or `VLMConversationDataset` directly (user chose to create RL2-native dataset)
- Cannot use: Context parallelism (CP=1 as specified in the draft)
- Fixed: Python path is `Megatron-Bridge/.venv/bin/python`; PYTHONPATH must include RL2 root

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The integration follows a layered approach:

```
Layer 1: Data (AudioSFTDataset)
  - Load HF dataset with audio column
  - Use Qwen2AudioProcessor to: apply_chat_template -> tokenize text -> extract mel features
  - Build labels with IGNORE_INDEX masking for non-assistant tokens
  - Collate: pad input_ids, attention_mask, input_features, feature_attention_mask, labels

Layer 2: Model Forward (MegatronWorker extension)
  - In _forward_step(), detect audio keys in minibatch
  - Pass input_features and feature_attention_mask as extra kwargs to model()
  - Qwen2AudioModel.forward() handles: embed text -> encode audio -> project -> merge -> LM forward

Layer 3: Training Loop (AudioSFTTrainer)
  - Similar to SFTTrainer but also initializes AutoProcessor
  - Passes processor to AudioSFTDataset for audio feature extraction
  - scatter_data distributes audio tensors along with text tensors

Layer 4: Configuration (YAML + Hydra)
  - Extends sft.yaml pattern with audio-specific fields
  - Freeze flags, audio processor path, dataset columns

Layer 5: Script (qwen_audio_sft.sh)
  - convert_checkpoints.py import (HF -> Megatron)
  - torchrun -m RL2.trainer.audio_sft (with Hydra overrides)
  - export_hf.py (Megatron -> HF)
```

**Key implementation detail for AudioSFTDataset**: The dataset should follow the pattern in `qwen_audio_sft/train_hf.py`'s `Qwen2AudioCollator` and Megatron-Bridge's `qwen2_audio_collate_fn`, but adapted to produce RL2's expected tensor_dict format (`states`, `actions`, `action_mask` for text, plus `input_features`, `feature_attention_mask` for audio). The label masking approach (find assistant span by backward search in tokenized input_ids) should be reused.

**Key implementation detail for forward pass**: In `MegatronWorker._forward_step()`, the current code passes `input_ids=minibatch["states"]` to the model. For audio, we additionally pass `input_features=minibatch["input_features"]` and `feature_attention_mask=minibatch["feature_attention_mask"]`. The `Qwen2AudioModel.forward()` already handles these correctly - it embeds the text, encodes the audio, projects it, merges audio embeddings into text embeddings at `<|AUDIO|>` token positions, then runs the language model.

**Key implementation detail for scatter_data**: The existing `scatter_data` and `_tensor_dict_to_minibatches` in `RL2/utils/sequences.py` assume 2D tensors `[batch, seq_len]`. Audio features are 3D `[batch, n_mels, mel_seq_len]`. The packing/distribution logic needs to handle these differently - audio features should be padded/concatenated along the batch dimension without sequence-level packing. Since CP=1, `slide_along_cp` is not applied, which simplifies the handling.

### Relevant References

- `RL2/RL2/datasets/sft.py` - Existing text SFTDataset, template for new AudioSFTDataset
- `RL2/RL2/datasets/base.py` - BaseDataset, get_dataloaders(), pack_tensor_dicts(), collate_fn pattern
- `RL2/RL2/workers/megatron/base.py` - MegatronWorker._forward_step() that needs extension
- `RL2/RL2/workers/megatron/actor.py` - MegatronActor.sft_step() and _scatter_data()
- `RL2/RL2/trainer/sft.py` - SFTTrainer, template for AudioSFTTrainer
- `RL2/RL2/trainer/config/sft.yaml` - Hydra config template
- `RL2/RL2/trainer/config/actor/megatron.yaml` - Actor config with model/optimizer settings
- `RL2/examples/limo_sft.sh` - Example script pattern to follow
- `Megatron-Bridge/src/megatron/bridge/data/vlm_datasets/collate.py` - `qwen2_audio_collate_fn()` reference for audio collation and label masking
- `Megatron-Bridge/src/megatron/bridge/data/vlm_datasets/hf_dataset_makers.py` - `make_default_audio_dataset()` reference for dataset formatting
- `Megatron-Bridge/src/megatron/bridge/models/qwen_audio/modeling_qwen2_audio.py` - `Qwen2AudioModel.forward()` to understand expected input format
- `Megatron-Bridge/src/megatron/bridge/models/qwen_audio/qwen2_audio_provider.py` - Provider with freeze support
- `Megatron-Bridge/src/megatron/bridge/training/audio_lm_step.py` - Forward step reference showing how audio inputs are passed to model
- `Megatron-Bridge/examples/conversion/convert_checkpoints.py` - Checkpoint conversion CLI
- `Megatron-Bridge/examples/models/audio_lm/qwen2_audio/export_hf.py` - Megatron-to-HF export
- `Megatron-Bridge/examples/models/audio_lm/qwen2_audio/conf/qwen2_audio_override_example.yaml` - YAML config reference
- `qwen_audio_sft/train_hf.py` - HF training reference, especially `Qwen2AudioCollator` and `build_conversation()`

## Dependencies and Sequence

### Milestones

1. **AudioSFTDataset**: Create the RL2-native audio dataset class
   - Phase A: Implement `AudioSFTDataset` class extending `BaseDataset` with audio loading, conversation formatting (using `build_conversation` pattern), and `Qwen2AudioProcessor`-based tokenization and mel feature extraction
   - Phase B: Implement `audio_collate_fn` that produces batched tensors: `input_ids`, `attention_mask`, `input_features`, `feature_attention_mask`, `labels`, `loss_mask`, adapted into RL2's `states`/`actions`/`action_mask` format plus audio keys
   - Phase C: Adapt `get_dataloaders` (or create `get_audio_dataloaders`) to work with the new dataset class

2. **Forward Pass Extension**: Extend MegatronWorker to support audio inputs
   - Phase A: Modify `_forward_step()` in `MegatronWorker` to detect and pass `input_features` and `feature_attention_mask` from minibatch to the model's forward call
   - Phase B: Ensure `scatter_data` / `_tensor_dict_to_minibatches` handles 3D audio tensors correctly (padding along batch dimension without sequence-packing audio features)

3. **YAML Configuration**: Create audio SFT config files
   - Phase A: Create `RL2/RL2/trainer/config/audio_sft.yaml` with audio-specific defaults (dataset columns, prompt, freeze flags, processor path)
   - Phase B: Create or extend `RL2/RL2/trainer/config/actor/megatron.yaml` with audio-related fields if needed (freeze flags)

4. **AudioSFTTrainer**: Create the audio SFT training entry point
   - Phase A: Create `RL2/RL2/trainer/audio_sft.py` with `AudioSFTTrainer` that extends `SFTTrainer`, initializes `AutoProcessor`, passes it to `AudioSFTDataset`
   - Phase B: Verify the training loop works end-to-end with the audio dataset and extended forward pass

5. **Shell Script and Checkpoint Conversion**: Create the orchestration script
   - Phase A: Create `RL2/examples/qwen_audio_sft.sh` with HF-to-Megatron conversion (using `convert_checkpoints.py import`), training launch (via `torchrun`), and Megatron-to-HF export (using `export_hf.py`)
   - Phase B: Make all paths and hyperparameters configurable via shell variables

**Dependency graph:**
- Milestone 2 (Forward Pass) is independent and can be done in parallel with Milestone 1 (Dataset)
- Milestone 3 (Config) is independent and can be done in parallel
- Milestone 4 (Trainer) depends on Milestones 1, 2, and 3
- Milestone 5 (Script) depends on Milestone 4

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Additional Technical Notes
- CP=1 simplifies the implementation significantly: no need to modify `slide_along_cp()` for audio features, no context-parallel slicing of audio tensors
- The `Qwen2AudioProcessor` from HuggingFace handles mel spectrogram extraction internally - no need to implement custom audio feature extraction
- The `AutoBridge.from_hf_pretrained()` automatically detects Qwen2-Audio architecture, so model loading in RL2 requires no special handling beyond using the correct model name
- Audio encoder and projector use HuggingFace implementations (not Megatron-ified); their gradients are synced across TP ranks via `hook_hf_module_setattr_for_tp_grad_sync()` in Megatron-Bridge
- The `Qwen2AudioModelProvider.provide()` handles freeze logic internally when `freeze_audio_model`, `freeze_audio_projection`, or `freeze_language_model` flags are set - the YAML config just needs to pass these flags through to the provider

--- Original Design Draft Start ---

@/workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/sft.sh 是使用 Megatron-Bridge 对 Qwen2-Audio-7B 用 aishell recipe 训练的脚本，默认使用的是 /workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/finetune_qwen2_audio.py

@/workspace_yuekai/asr/RL2/examples/limo_sft.sh 是使用现在的 RL2 框架，训练文本 LLM 的方法，想成功运行，需要使用 export PYTHONPATH=/workspace_yuekai/asr/RL2:$PYTHONPATH
python_path=/workspace_yuekai/asr/Megatron-Bridge/.venv/bin/python

@/workspace_yuekai/asr/qwen_audio_sft/run.sh 是使用 huggingface 代码训练 qwen2 audio 的 recipe, 你可以看看 @/workspace_yuekai/asr/qwen_audio_sft/train_hf.py 和数据构造相关的部分


我希望你能在现在的 RL2 框架中，帮我集成使用 Megatron 训练 像 qwen-audio 这样的 audio llm 的代码，最后给我一个 @/workspace_yuekai/asr/RL2/examples/qwen_audio_sft.sh

Note：
训练的 CP=1， 你可以不用考虑和 CP 有关的部分

huggingface -> megatron checkpont 以及 megatron checkpoint -> hf 的脚本，帮我从这两个地方 /workspace_yuekai/asr/Megatron-Bridge/examples/conversion/convert_checkpoints.py, 以及 /workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/export_hf.py

我希望能像 /workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/conf/qwen2_audio_override_example.yaml 类似这样，有一个 yaml 能管理我的训练配置
--- Original Design Draft End ---
