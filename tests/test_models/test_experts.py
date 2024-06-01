#import sys
#sys.path.append("/home/ketkar/Public/Python_programs/DOE")
#import unittest
#from src.models.experts import Gate, TempConfig, ExpertsModule, Expert

#class TestExperts(unittest.TestCase):
#    def setUp(self) -> None:
#        self.intermediate_size = 5
#        self.hidden_size = 5
#        self.config = TempConfig(intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, num_local_experts=self.hidden_size)
#        self.gate = Gate(config=self.config)
#    
#    def test_gating():
#        pass
#
#    def test_experts_module():
#        pass


from sample_experts import Gate, Expert, ExpertsModule, TempConfig
import torch

batch_size = 5
seq_len = 2
num_local_experts = 4
hidden_size = 3
intermediate_size = 3

config = TempConfig(intermediate_size=intermediate_size, hidden_size=hidden_size, num_local_experts=num_local_experts)

gating_indices = torch.tensor([[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]])
hidden_states = torch.randn(batch_size, seq_len, 3)
print(f"Gating indices:")
print(gating_indices)
print("\n")
print(f"Hidden states:")
print(hidden_states)
print("\n")
EM = ExpertsModule(config)
output: torch.Tensor = EM(hidden_states, gating_indices)
print("Output: ")
print(output)