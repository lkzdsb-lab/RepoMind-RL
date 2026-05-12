from baseregister import BaseRegistry
from model.prompt import PromptSpec

class PromptRegistry(BaseRegistry):
    def register(self, spec: PromptSpec) -> None:
        super().register(spec)