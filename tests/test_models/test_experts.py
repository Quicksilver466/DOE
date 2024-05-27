import unittest
from src.models.experts import TempConfig, Gate, Expert, ExpertsModule
import torch
from torch.fx import GraphModule, symbolic_trace, replace_pattern

def replace_linear(module: GraphModule) -> GraphModule:
    pass # Replace all linear opertations with identity operations

class TestExperts(unittest.TestCase):
    def setUp(self) -> None:
        self.intermediate_size = 5
        self.hidden_size = 5
        self.config = TempConfig(intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, num_local_experts=self.hidden_size)
        self.gate = Gate(config=self.config)
    
    def test_gating():
        pass

    def test_experts_module():
        pass