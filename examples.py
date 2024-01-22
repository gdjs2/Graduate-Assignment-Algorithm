import GA
import torch
from loguru import logger
from rich.console import Console
from numba import jit

NODES_NUM = 10

def c(x: torch.Tensor, y: torch.Tensor):
    return x * y

def hueristic(M: torch.Tensor):
    max_values, _ = torch.max(M, dim=0)
    return M.eq(max_values).int()
    
if __name__ == '__main__':
    console = Console()
    logger.remove()
    logger.add(console.log)

    M_gt = torch.zeros((NODES_NUM, NODES_NUM)).to('mps')
    M_gt[torch.arange(0, NODES_NUM, dtype=torch.int64), torch.randperm(NODES_NUM)] = 1

    G = torch.rand(NODES_NUM, NODES_NUM).to('mps')
    G = (G + G.t() > 1.3).float()
    torch.diagonal(G).fill_(0)
    g = torch.mm(torch.mm(M_gt.t(), G), M_gt)

    eps = .5
    beta_0 = .5
    beta_f = 10
    beta_r = 1.075
    I0 = 4
    I1 = 30

    M = GA.GA(G, g, c, beta_0, beta_r, beta_f, eps, I0, I1, logger=logger)
    result = hueristic(M[:-1, :-1])

    print(result)
    print(M_gt)
    print(torch.equal(result, M_gt))
