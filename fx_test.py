import torch
import torch.fx as fx
from torch.fx import symbolic_trace
import time
import re

def transform_test(module: torch.nn.Module, replacement_module: torch.nn.Module) -> torch.nn.Module:
    nn_graph: fx.GraphModule = symbolic_trace(module)

    for node in nn_graph.graph.nodes:
        if(node.op == "call_module" and re.search(r"linear", node.target)): node.target = replacement_module

class LinLayers(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear1(x)
        output = self.linear2(output)

        return output

class SingLinLayer(torch.nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)
        self.state = None

    def forward(self, x: torch.Tensor):
        self.state = x.sum(dim=0)
        return self.linear(x)

class NN_mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = SingLinLayer(10, 5)
        self.linear2 = SingLinLayer(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear1(x)
        output = self.linear2(output)

        return output

tic = time.time()

nn = LinLayers()
my_lin = SingLinLayer(10, 5)
nn_mod = NN_mod()

nn_graph: fx.GraphModule = symbolic_trace(nn)
lin_graph: fx.GraphModule = symbolic_trace(my_lin)
mod_graph: fx.GraphModule = symbolic_trace(nn_mod)



#for node in nn_graph.graph.nodes:
#    print(node.args)

nn_graph.graph.print_tabular()
print("\n")
lin_graph.graph.print_tabular()
print("\n")
mod_graph.graph.print_tabular()

toc = time.time()
print(f"\nIt took {toc - tic} seconds to execute above code")