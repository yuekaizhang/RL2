export PYTHONPATH=/workspace_yuekai/asr/RL2:$PYTHONPATH
python_path=/workspace_yuekai/asr/Megatron-Bridge/.venv/bin/python

${python_path} -m torch.distributed.run \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.train.path=Chenmien/LIMO \
    data.train.max_length=16384 \
    data.train.batch_size=32 \
    actor.model_name=Qwen/Qwen2.5-0.5B-Instruct \
    actor.tf_config.context_parallel_size=4 \
    actor.max_length_per_device=4096 \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-7b-inst \
    trainer.n_epochs=15
