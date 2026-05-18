from baseregister import BaseRegistry
from model.skill import SkillSpec

class SkillRegistry(BaseRegistry):
    def register(self, spec: SkillSpec) -> None:
        super().register(spec)