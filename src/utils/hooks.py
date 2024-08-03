from torch.nn import Module
from typing import Callable

class HookRegistry():
    def __init__(self) -> None:
        self.registry: dict[Module, Callable] = {}

    def add_hook(self, layer: Module, func: Callable):
        self.registry[layer] = func

    def register_hooks(self):
        for module, callback in self.registry.items():
            module.register_forward_hook(callback)