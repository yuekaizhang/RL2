import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor
from RL2.trainer import Trainer
from RL2.datasets.audio_sft import get_audio_dataloaders
from RL2.workers import initialize_actor
from RL2.utils.communication import initialize_global_process_group


# Keys that are included in collate_fn output for AC-1 verification
# but must be filtered before entering the RL2 training pipeline
# (they have full seq_len shape, incompatible with the RL2 S-1 convention)
_AC1_VERIFICATION_KEYS = {"input_ids", "attention_mask", "labels", "loss_mask"}


class AudioSFTTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)

        self.processor = AutoProcessor.from_pretrained(
            config.actor.model_name, trust_remote_code=True
        )

        self.train_dataloader, self.test_dataloader = get_audio_dataloaders(
            config.data, self.processor
        )
        self.actor.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )

    @staticmethod
    def _filter_for_training(tensor_dict):
        """Remove AC-1 verification keys before passing to the training pipeline."""
        return {
            k: v for k, v in tensor_dict.items()
            if k not in _AC1_VERIFICATION_KEYS
        }

    def train(self):

        step = self.load_ckpt((self.actor,))
        for epoch in range(
            step // len(self.train_dataloader),
            self.config.trainer.n_epochs
        ):
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):

                step += 1
                self.actor.sft_step(
                    self._filter_for_training(tensor_dict), True, step
                )
                self.save_ckpt((self.actor,), step)

            for tensor_dict in self.test_dataloader:
                self.actor.sft_step(
                    self._filter_for_training(tensor_dict), False, step
                )

        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="audio_sft", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group()

    trainer = AudioSFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
