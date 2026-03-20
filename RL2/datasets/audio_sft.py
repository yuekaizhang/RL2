import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoProcessor

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

# Default audio parameters for Qwen2-Audio (Whisper-based feature extractor)
_DEFAULT_MAX_AUDIO_SECONDS = 30
_DEFAULT_SAMPLING_RATE = 16000


def build_conversation(
    prompt: str, answer: str, remove_text_spaces: bool = True
) -> List[Dict]:
    if remove_text_spaces:
        answer = answer.replace(" ", "")
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]


def _validate_audio(audio_raw, idx: int) -> Tuple[np.ndarray, int]:
    """Validate and extract audio array from various HF audio formats.

    Returns (audio_array, sampling_rate).
    Raises ValueError with a descriptive message for missing or corrupted audio.
    """
    if audio_raw is None:
        raise ValueError(
            f"Example {idx}: audio payload is None. "
            "The audio file may be missing or failed to load."
        )

    sampling_rate = _DEFAULT_SAMPLING_RATE

    if isinstance(audio_raw, dict):
        arr = audio_raw.get("array")
        if arr is None:
            raise ValueError(
                f"Example {idx}: audio dict has no 'array' key. "
                f"Available keys: {list(audio_raw.keys())}"
            )
        if "sampling_rate" in audio_raw:
            sampling_rate = audio_raw["sampling_rate"]
    elif isinstance(audio_raw, (tuple, list)):
        arr = audio_raw[0]
        if len(audio_raw) > 1:
            sampling_rate = audio_raw[1]
    else:
        arr = audio_raw

    if isinstance(arr, np.ndarray):
        if arr.size == 0:
            raise ValueError(
                f"Example {idx}: audio array is empty (size=0)."
            )
    elif isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            raise ValueError(
                f"Example {idx}: audio array is empty (length=0)."
            )
    elif arr is None:
        raise ValueError(
            f"Example {idx}: extracted audio array is None."
        )

    return arr, sampling_rate


def _truncate_audio(
    audio_arr, sampling_rate: int, max_audio_seconds: float
) -> np.ndarray:
    """Truncate audio array to max_audio_seconds if it exceeds the limit.

    This ensures overlong audio is explicitly handled before processor
    invocation, aligned with the feature extractor's capacity (e.g.,
    Whisper's 30-second window).
    """
    max_samples = int(max_audio_seconds * sampling_rate)
    if hasattr(audio_arr, '__len__') and len(audio_arr) > max_samples:
        logger.info(
            f"Audio truncated from {len(audio_arr)} to {max_samples} samples "
            f"({max_audio_seconds}s at {sampling_rate}Hz)"
        )
        return audio_arr[:max_samples]
    return audio_arr


class AudioSFTDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        processor: AutoProcessor,
        dataset,
    ):
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dataset = dataset

        # Force right padding so that real tokens start at position 0.
        # The RL2 collator computes eos_mask assuming right-padded layout
        # (real tokens first, padding at the end). Qwen2-Audio's tokenizer
        # defaults to padding_side="left", which places padding at the start
        # and causes slide_along_cp to truncate actual audio tokens.
        self.tokenizer.padding_side = "right"

        # Validate required columns exist
        audio_column = getattr(config, "audio_column", "audio")
        text_column = getattr(config, "text_column", "text")
        if hasattr(dataset, "column_names"):
            columns = dataset.column_names
            if audio_column not in columns:
                raise ValueError(
                    f"Dataset is missing required audio column '{audio_column}'. "
                    f"Available columns: {columns}"
                )
            if text_column not in columns:
                raise ValueError(
                    f"Dataset is missing required text column '{text_column}'. "
                    f"Available columns: {columns}"
                )

        # Precompute special token IDs for boundary-aware label masking.
        # Qwen2 chat template uses <|im_start|> / <|im_end|> as turn markers.
        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Tokenize the assistant role header ("assistant\n") to compute the
        # number of tokens to skip after <|im_start|> to reach content.
        self._assistant_header_ids = self.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

    def _find_assistant_content_boundaries(
        self, ids: List[int]
    ) -> Tuple[int, int]:
        """Find the token boundaries of the assistant's content.

        Uses chat template structural markers (<|im_start|>, <|im_end|>)
        instead of text-based substring search. This is deterministic and
        does not depend on tokenization of the assistant text matching.

        Returns (content_start, content_end) token indices, or (-1, -1) if
        the boundaries cannot be determined.
        """
        # Find the last <|im_start|> — in our single-turn format, this is
        # always the assistant turn (system, user, assistant).
        last_im_start = -1
        for j in range(len(ids) - 1, -1, -1):
            if ids[j] == self._im_start_id:
                last_im_start = j
                break

        if last_im_start < 0:
            return -1, -1

        # Verify the role header matches "assistant\n"
        header_start = last_im_start + 1
        header_end = header_start + len(self._assistant_header_ids)

        if header_end > len(ids):
            return -1, -1

        actual_header = ids[header_start:header_end]
        if actual_header != self._assistant_header_ids:
            return -1, -1

        content_start = header_end

        # Find the next <|im_end|> after the content start
        content_end = len(ids)
        for j in range(content_start, len(ids)):
            if ids[j] == self._im_end_id:
                content_end = j
                break

        return content_start, content_end

    def collate_fn(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        conversations = []
        audios = []

        prompt = getattr(self.config, "prompt", "Transcribe the audio clip.")
        audio_column = getattr(self.config, "audio_column", "audio")
        text_column = getattr(self.config, "text_column", "text")
        remove_text_spaces = getattr(self.config, "remove_text_spaces", True)
        max_length = getattr(self.config, "max_length", 4096)
        max_audio_seconds = getattr(
            self.config, "max_audio_seconds", _DEFAULT_MAX_AUDIO_SECONDS
        )

        for idx, ex in enumerate(examples):
            answer = ex[text_column]
            conv = build_conversation(prompt, answer, remove_text_spaces)

            try:
                audio_arr, sr = _validate_audio(ex[audio_column], idx)
            except ValueError as e:
                warnings.warn(str(e) + " Skipping this example.")
                continue

            # Explicit audio truncation aligned with feature extractor capacity
            audio_arr = _truncate_audio(audio_arr, sr, max_audio_seconds)

            conversations.append(conv)
            audios.append(audio_arr)

        if not conversations:
            raise RuntimeError(
                "All examples in this batch had invalid audio. Cannot proceed."
            )

        texts = [
            self.processor.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        # Route truncation/max_length only to the tokenizer via text_kwargs.
        # Passing them as flat kwargs would leak to WhisperFeatureExtractor
        # (both TextKwargs and AudioKwargs declare max_length/truncation),
        # causing mel features to be padded to 4096 samples (~25 frames)
        # instead of the default 480000 (3000 frames) expected by the
        # Whisper-based Qwen2AudioEncoder.
        batch = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            text_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": max_length,
            },
        )

        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        # Build labels using boundary-aware masking via structural tokens.
        # Start with all tokens masked, then unmask only assistant content.
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            content_start, content_end = self._find_assistant_content_boundaries(
                ids
            )
            if content_start >= 0 and content_start < content_end:
                labels[i, content_start:content_end] = input_ids[
                    i, content_start:content_end
                ]
            else:
                warnings.warn(
                    f"Example {i}: could not determine assistant content "
                    f"boundaries via chat template tokens. All tokens masked."
                )

        # Shift labels for next-token prediction (Megatron convention)
        shifted_labels = labels[:, 1:]
        shifted_labels = torch.cat(
            [shifted_labels, IGNORE_INDEX * torch.ones_like(shifted_labels[:, :1])],
            dim=1,
        )

        # Derive loss_mask from shifted labels
        loss_mask = (shifted_labels != IGNORE_INDEX).float()

        # Build RL2-compatible tensor_dict
        states = input_ids[:, :-1]
        actions = input_ids[:, 1:]
        action_mask = loss_mask[:, :-1]

        # Compute actual sequence lengths from attention_mask and set eos_mask
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        seq_lengths = attention_mask.sum(dim=1)
        eos_mask = torch.zeros_like(states)
        for i in range(batch_size):
            eos_pos = min(seq_lengths[i].item() - 2, states.shape[1] - 1)
            eos_pos = max(0, int(eos_pos))
            eos_mask[i, eos_pos] = 1

        tensor_dict = {
            # RL2 training tensors
            "states": states,
            "actions": actions,
            "action_mask": action_mask,
            "eos_mask": eos_mask,
            # Full-sequence diagnostic tensors for verification
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": shifted_labels,
            "loss_mask": loss_mask,
        }

        # Add audio-specific tensors
        if "input_features" in batch:
            tensor_dict["input_features"] = batch["input_features"]
        if "feature_attention_mask" in batch:
            tensor_dict["feature_attention_mask"] = batch["feature_attention_mask"]

        return tensor_dict


def get_audio_dataloaders(
    config: DictConfig,
    processor: AutoProcessor,
    batch_size: int = None,
) -> Tuple:
    from RL2.datasets.base import StatefulCycleDataLoader

    def _load_dataset(
        dataset_name: str,
        dataset_subset: Optional[str],
        dataset_split: str,
    ):
        if dataset_subset:
            return load_dataset(dataset_name, dataset_subset, split=dataset_split)
        else:
            return load_dataset(dataset_name, split=dataset_split)

    train_dataset = _load_dataset(
        config.train.dataset_name,
        getattr(config.train, "dataset_subset", None),
        getattr(config.train, "dataset_split", "train"),
    )

    if getattr(config.test, "dataset_name", None):
        test_dataset = _load_dataset(
            config.test.dataset_name,
            getattr(config.test, "dataset_subset", None),
            getattr(config.test, "dataset_split", "test"),
        )
    else:
        total_size = len(train_dataset)
        indices = np.arange(total_size)
        np.random.seed(42)
        np.random.shuffle(indices)
        test_ratio = getattr(config, "test_ratio", 0.03)
        split_point = int(test_ratio * total_size)
        train_indices, test_indices = indices[split_point:], indices[:split_point]
        test_dataset = train_dataset.select(test_indices)
        train_dataset = train_dataset.select(train_indices)

    train_dataset = AudioSFTDataset(config.train, processor, train_dataset)
    test_dataset = AudioSFTDataset(config.test, processor, test_dataset)

    def _get_dataloader(dataset: AudioSFTDataset, bs: int):
        return StatefulCycleDataLoader(
            dataset=dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            collate_fn=dataset.collate_fn,
        )

    train_dataloader = _get_dataloader(
        train_dataset, batch_size or config.train.batch_size
    )
    test_dataloader = _get_dataloader(
        test_dataset, batch_size or len(test_dataset)
    )
    return train_dataloader, test_dataloader
