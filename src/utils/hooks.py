from torch.nn import Module
from typing import Callable
from torch import Tensor
import logging

INFO_LOGGER = logging.getLogger("DOE-Info")

class HookRegistry():
    def __init__(self, model: Module) -> None:
        self.forward_registry: dict[Module, Callable] = {}
        self.backward_registry: dict[Module, Callable] = {}
        #self.add_all_hooks(model=model)

    def add_hook(self, module: Module, func: Callable):
        self.registry[module] = func

    def register_hooks(self):
        for module, callback in self.forward_registry.items():
            module.register_forward_hook(callback)

        for module, callback in self.backward_registry.items():
            module.register_full_backward_hook(callback)

    def add_all_hooks(self, model: Module):
        self.forward_registry.update({model.gating_model.gate: linear_param_forwardhook})
        self.forward_registry.update({model.gating_model.gate: linear_input_forwardhook})
        self.backward_registry.update({model.gating_model.gate: linear_grad_backwardhook})

def linear_param_forwardhook(module: Module, input: Tensor, output: Tensor):
    INFO_LOGGER.info(f"\nThe linear layer params are: \n")
    for name, param in module.named_parameters():
        INFO_LOGGER.info(f"{name}: {param}\n")

def linear_input_forwardhook(module: Module, input: Tensor, output: Tensor):
    INFO_LOGGER.info(f"The input for Gate Linear is: \n{input}\n")

def linear_grad_backwardhook(module: Module, grad_input: Tensor, grad_output: Tensor):
    INFO_LOGGER.info(f"The grad inputs are: \n{grad_input}\n")
    INFO_LOGGER.info(f"The parameter grads are: \n")
    for name, param in module.named_parameters():
        INFO_LOGGER.info(f"{name}: {param.grad}\n")