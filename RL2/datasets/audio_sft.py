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


def _validate_audio(audio_raw, idx: int) -> np.ndarray:
    """Validate and extract audio array from various HF audio formats.

    Raises ValueError with a descriptive message for missing or corrupted audio.
    """
    if audio_raw is None:
        raise ValueError(
            f"Example {idx}: audio payload is None. "
            "The audio file may be missing or failed to load."
        )

    if isinstance(audio_raw, dict):
        arr = audio_raw.get("array")
        if arr is None:
            raise ValueError(
                f"Example {idx}: audio dict has no 'array' key. "
                f"Available keys: {list(audio_raw.keys())}"
            )
    elif isinstance(audio_raw, (tuple, list)):
        arr = audio_raw[0]
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

    return arr


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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

    def collate_fn(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        conversations = []
        audios = []
        assistant_texts = []

        prompt = getattr(self.config, "prompt", "Transcribe the audio clip.")
        audio_column = getattr(self.config, "audio_column", "audio")
        text_column = getattr(self.config, "text_column", "text")
        remove_text_spaces = getattr(self.config, "remove_text_spaces", True)
        max_length = getattr(self.config, "max_length", 4096)

        valid_examples = []
        for idx, ex in enumerate(examples):
            answer = ex[text_column]
            conv = build_conversation(prompt, answer, remove_text_spaces)
            clean_answer = answer.replace(" ", "") if remove_text_spaces else answer

            try:
                audio_arr = _validate_audio(ex[audio_column], idx)
            except ValueError as e:
                warnings.warn(str(e) + " Skipping this example.")
                continue

            conversations.append(conv)
            assistant_texts.append(clean_answer)
            audios.append(audio_arr)
            valid_examples.append(ex)

        if not conversations:
            raise RuntimeError(
                "All examples in this batch had invalid audio. Cannot proceed."
            )

        texts = [
            self.processor.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        batch = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        # Build labels: mask all non-assistant tokens with IGNORE_INDEX
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, assistant_text in enumerate(assistant_texts):
            ids = input_ids[i].tolist()
            assistant_token_ids = self.tokenizer(
                assistant_text, add_special_tokens=False
            )["input_ids"]

            span_len = len(assistant_token_ids)
            found = -1
            # Search backward to find the last occurrence of the assistant span
            for start in range(len(ids) - span_len, -1, -1):
                if ids[start: start + span_len] == assistant_token_ids:
                    found = start
                    break

            if found >= 0:
                # Only unmask the assistant content tokens
                labels[i, found: found + span_len] = input_ids[i, found: found + span_len]
            else:
                warnings.warn(
                    f"Could not find assistant span for example {i}, masking all"
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
            # AC-1 required tensors for verification/debugging
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
