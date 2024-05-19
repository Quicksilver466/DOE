import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class TempConfig:
    intermediate_size: int
    hidden_size: int
    num_local_experts: int = 0

class MLP(nn.Module):
    def __init__(self, config: TempConfig) -> None:
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.ReLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class TaskExperts(nn.Module):
    def __init__(self, config: TempConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.gates = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        self.sig_func = nn.Sigmoid()
        self.expert_indices = []
        self.threshold = 0.5

    def forward(self, hidden_states: torch.Tensor, decide_experts=False) -> torch.Tensor:
        if decide_experts: self.reset_indices(hidden_states)
        for index in self.expert_indices:
            output = self.experts[index](hidden_states)

        return output

    def reset_indices(self, hidden_states: torch.Tensor) -> None:
        cls_state = hidden_states[..., 0, :]
        gating_logits = self.gates(cls_state)
        gating_logits = self.sig_func(gating_logits)
        pass