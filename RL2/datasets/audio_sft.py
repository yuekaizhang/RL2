import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

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

        for ex in examples:
            answer = ex[text_column]
            conv = build_conversation(prompt, answer, remove_text_spaces)
            conversations.append(conv)
            clean_answer = answer.replace(" ", "") if remove_text_spaces else answer
            assistant_texts.append(clean_answer)

            audio = ex[audio_column]
            if isinstance(audio, dict):
                audios.append(audio["array"])
            elif isinstance(audio, tuple):
                audios.append(audio[0])
            else:
                audios.append(audio)

        texts = [
            self.processor.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        batch = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        # Build labels: mask non-assistant tokens with IGNORE_INDEX
        labels = input_ids.clone()
        for i, assistant_text in enumerate(assistant_texts):
            ids = input_ids[i].tolist()
            assistant_token_ids = self.tokenizer(
                assistant_text, add_special_tokens=False
            )["input_ids"]

            span_len = len(assistant_token_ids)
            found = -1
            for start in range(len(ids) - span_len, -1, -1):
                if ids[start: start + span_len] == assistant_token_ids:
                    found = start
                    break

            if found >= 0:
                labels[i, :found] = IGNORE_INDEX
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is not None:
                    labels[i][input_ids[i] == pad_token_id] = IGNORE_INDEX
            else:
                warnings.warn(
                    f"Could not find assistant span for example {i}, masking all"
                )
                labels[i, :] = IGNORE_INDEX

        # Shift labels for next-token prediction (Megatron convention)
        shifted_labels = labels[:, 1:]
        shifted_labels = torch.cat(
            [shifted_labels, IGNORE_INDEX * torch.ones_like(shifted_labels[:, :1])],
            dim=1,
        )

        # Derive loss_mask and action_mask from shifted labels
        loss_mask = (shifted_labels != IGNORE_INDEX).float()

        # Build RL2-compatible tensor_dict
        # states = input_ids[:, :-1], actions = input_ids[:, 1:]
        states = input_ids[:, :-1]
        actions = input_ids[:, 1:]
        action_mask = loss_mask[:, :-1]

        # Compute actual sequence lengths from attention_mask and set eos_mask
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        seq_lengths = attention_mask.sum(dim=1)  # actual lengths (before shift)
        eos_mask = torch.zeros_like(states)
        for i in range(batch_size):
            eos_pos = min(seq_lengths[i].item() - 2, states.shape[1] - 1)
            eos_pos = max(0, int(eos_pos))
            eos_mask[i, eos_pos] = 1

        tensor_dict = {
            "states": states,
            "actions": actions,
            "action_mask": action_mask,
            "eos_mask": eos_mask,
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
    import numpy as np

    audio_column = getattr(config.train, "audio_column", "audio")
    text_column = getattr(config.train, "text_column", "text")

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
