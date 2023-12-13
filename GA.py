import torch
import typing
import logging
import pygmtools as pygm

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def differential(
        M: torch.Tensor, 
        G: torch.Tensor, 
        g: torch.Tensor, 
        c: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    """
    Differential M_{ai} on E_{wg}
    
    Args:
        M: matching matrix
        G: graph 1
        g: graph 2
        c: value function
    
    Return:
        Q: differential matrix
    """
    A = G.size(0)
    I = g.size(0)
    Q = torch.zeros_like(M).to(M.device)

    # Here is extremly complex, how to improve?
    for a in range(A):
        for i in range(I):
            for b in range(A):
                for j in range(I):
                    if G[a, b] != 0 and g[i, j] != 0:
                        Q[a, i] += M[b, j] * c(G[a, b], g[i, j])
    
def GA(
        G: torch.Tensor, 
        g: torch.Tensor, 
        c: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        beta_0: float, 
        beta_r: float, 
        beta_f: float, 
        eps: float, 
        I0: int, 
        I1: int,
        device: torch.device = None,
        logger: logging.Logger = None
):
    """
    Graduate Assignment Algorithm
    
    Args:
        G: graph 1
        g: graph 2
        c: value function
        beta_0: initial value of the control parameter beta
        beta_r: rate at which the control parameter beta is increased
        beta_f: maximum value of the control parameter beta
        eps: epsilon
        I0: maximum # of iterations allowed at each value of the control parameter beta
        I1: maximum # of iterations allowed for Sinkhorn's method
        logger: logger, there will be no logging if logger is None
    
    Return:
        M: matching matrix
    """

    device = get_device() if device is None else device
    pygm.BACKEND = 'pytorch'

    logger.info(
f'''Graduate Assignment Algorithm:
        G: {G.shape}
        g: {g.shape}
        beta_0: {beta_0}
        beta_r: {beta_r}
        beta_f: {beta_f}
        eps: {eps}
        I0: {I0}
        I1: {I1}
        device: {device}
'''
    ) if logger is not None else None

    beta = beta_0

    M = torch.ones((G.size(0)+1, g.size(0)+1)).to(device)
    G = G.to(device)
    g = g.to(device)

    while beta < beta_f:
        M_old = M.clone()
        for i in range(I0):
            # Q_{ai} \leftarrow -\frac{\partial E_{wg}}{\partial M_{a_i}}
            Q = differential(M[:-1, :-1], G, g, c)
            # M_{ai}^0 \leftarrow \exp(\beta Q_{ai})
            # Because in pygm.sinkhorn, it will perform exponential on all elements.
            # To avoid exponential on the slack row & colmn, we perform log here to counteract it.
            M[:-1, :-1] = beta * Q
            M[:-1, -1] = torch.log(M[:-1, -1])
            M[-1, :-1] = torch.log(M[-1, :-1])
            M[-1, -1] = torch.log(M[-1, -1])

            # sinkhorn
            M = pygm.sinkhorn(M, max_iter=I1)
            # break if M converges
            if torch.sum(torch.abs(M[:-1, :-1] - M_old[:-1, :-1])) < eps: break
            M_old = M.clone()
        
        logger.info(f'beta: {beta}, i: {i} iterations') if logger is not None else None
        beta *= beta_r
    
    return M


