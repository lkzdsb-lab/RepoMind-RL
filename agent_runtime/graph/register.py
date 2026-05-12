from baseregister import BaseRegistry
from model.agent.node import NodeSpec

class NodeRegistry(BaseRegistry):
    def register(self, spec: NodeSpec) -> None:
        super().register(spec)