from sgl_jax.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2Model


class MiMoModel(Qwen2Model):
    pass


class MiMoForCausalLM(Qwen2ForCausalLM):
    pass

EntryClass = MiMoForCausalLM

