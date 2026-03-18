from typing import Dict, Optional, Tuple, List
from omegaconf import DictConfig
import torch
from megatron.core import parallel_state as mpu
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.logging import (
    time_logger,
    gather_and_log,
)


class MegatronActor(MegatronWorker):

    def __init__(self, config: DictConfig, train: bool):
        super().__init__(config, train)

        self.model = self.provider.provide_distributed_model(
            ddp_config=self.ddp_config,
            wrap_with_ddp=train
        )
        self._prepare_model_optimizer()

    @time_logger("update_actor")
    def sft_step(
        self,
        tensor_dict: Optional[Dict[str, torch.Tensor]],
        train: bool,
        step: int
    ):
        minibatches = self._scatter_data(tensor_dict)
        for model in self.model:
            model.train(train)

        total_actions, total_sequences = count_total(
            minibatches,
            ("action_mask", "eos_mask"),
            mpu.get_data_parallel_group()
        )

        def f(
            minibatch: Dict[str, torch.Tensor],
            cu_seqlens: torch.Tensor,
            logits: torch.Tensor,
            non_loss_data: bool = False
        ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:

            compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group()
            )
            minibatch = gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            suffix = "train" if train else "test"
            metric = {f"loss/{suffix}": [loss.item()]}
            return metric if non_loss_data else (self._scale_loss(loss), metric)

        with torch.set_grad_enabled(train):
            metrics = self._forward_backward(f, minibatches)
        if train:
            grad_norm = self._optimizer_step()
            metrics["grad_norm"] = [grad_norm]
        gather_and_log(metrics, step, mpu.get_data_parallel_group())
