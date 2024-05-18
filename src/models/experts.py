import torch
from torch import nn
from transformers.models import phi

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class TaskExperts(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts

        self.gates = nn.Linear(self.hidden_dim, self.num_experts)
        