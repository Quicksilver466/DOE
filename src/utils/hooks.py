from torch.nn import Module
from typing import Callable
from torch import Tensor
import mlflow

class HookRegistry():
    def __init__(self) -> None:
        self.registry: dict[Module, Callable] = {}

    def add_hook(self, layer: Module, func: Callable):
        self.registry[layer] = func

    def register_hooks(self):
        for module, callback in self.registry.items():
            module.register_forward_hook(callback)

def logitloss_hook(module: Module, input: Tensor, output: Tensor):
    input_list = input.tolist()
    output_list = output.tolist()
    mlflow.log_param("LogitLossInput", input_list)
    mlflow.log_param("LogitLossOutput", output_list)