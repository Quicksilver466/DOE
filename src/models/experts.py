import torch
from torch import nn, Tensor
from dataclasses import dataclass

@dataclass
class TempConfig:
    intermediate_size: int
    hidden_size: int
    num_local_experts: int = 1
    threshold: float = 0.5

class Gate(nn.Module):
    def __init__(self, config: TempConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.sig_func = nn.Sigmoid()
        self.threshold = config.threshold

    def forward(self, cls_hidden_states: Tensor) -> Tensor:
        gating_logits = self.gate(cls_hidden_states)
        gating_output = self.sig_func(gating_logits)
        gating_output = torch.where(gating_output > self.threshold, 1, 0)
        return gating_output

class Expert(nn.Module):
    def __init__(self, config: TempConfig) -> None:
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.ReLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class ExpertsModule(nn.Module):
    def __init__(self, config: TempConfig) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: Tensor, expert_indices: Tensor) -> Tensor:
        outputs = []
        expert_indices = expert_indices.permute(1, 0)
        for i, expert_index_tensor in enumerate(expert_indices):
            ones_indices = torch.nonzero(expert_index_tensor)
            expert_batch = torch.index_select(hidden_states, 0, ones_indices.view(-1))
            outputs.append(self.experts[i](expert_batch))

        return outputs