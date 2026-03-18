from .base import Worker

def initialize_actor(config, train):

    from .megatron.actor import MegatronActor
    return MegatronActor(config, train)
