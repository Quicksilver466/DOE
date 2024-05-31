import torch
from torch import nn
import torch.fx as fx
from torch.fx import symbolic_trace, replace_pattern
import time

class Pattern(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

class LinLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear1(x)
        output = self.linear2(output)

        return output

class SingLinLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.state = None

    def forward(self, x: torch.Tensor):
        self.state = x.sum(dim=0)
        return self.linear(x)

class NN_mod(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = SingLinLayer(10, 5)
        self.linear2 = SingLinLayer(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear1(x)
        output = self.linear2(output)

        return output

tic = time.time()

lin_lay = LinLayers()
sing_lin = SingLinLayer(10, 5)
nn_mod = NN_mod()

nn_graph: fx.GraphModule = symbolic_trace(lin_lay)
lin_graph: fx.GraphModule = symbolic_trace(sing_lin)
mod_graph: fx.GraphModule = symbolic_trace(nn_mod)

nn_graph.graph.print_tabular()
print("\n")
lin_graph.graph.print_tabular()
print("\n")
mod_graph.graph.print_tabular()
print("\n")

symbolic_trace(Pattern()).graph.print_tabular()
print("\n")

replace_pattern(nn_graph, Pattern(), sing_lin)
nn_graph.graph.print_tabular()

toc = time.time()
print(f"\nIt took {toc - tic} seconds to execute above code")