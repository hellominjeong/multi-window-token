import torch
from typing import Tuple, Callable
import math
import gc

gc.collect()
torch.cuda.empty_cache()  # 캐시에 남아있는 GPU 메모리 해제
torch.cuda.reset_peak_memory_stats()  # 메모리 사용량 기록 초기화
device = "cuda:2" if torch.cuda.is_available() else "cpu"
import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode: str = None):
    return x

import torch
from typing import Tuple, Callable

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def bipartite_soft_matching_global(metric: torch.Tensor,
                                 w: int, h: int, r: int,
                                 no_rand: bool = False,
                                 generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst globally and merges r tokens from src to dst.
    Instead of using spatial (sx, sy) regions, this version randomly selects dst tokens from the entire image.
    
    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use first tokens as dst)
     - generator: random number generator
    """
    B, N, _ = metric.shape
    
    if r <= 0:
        return lambda x: x, lambda x: x
        
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        # Number of destination tokens - let's use sqrt(N) as a reasonable default
        num_dst = int(torch.sqrt(torch.tensor(N)).item())
        
        # Randomly select dst tokens
        if no_rand:
            rand_idx = torch.arange(N, device=metric.device).reshape(1, -1, 1)
        else:
            rand_idx = torch.randperm(N, device=metric.device, generator=generator).reshape(1, -1, 1)
            
        # Split into src and dst indices
        b_idx = rand_idx[:, :num_dst, :] # dst
        a_idx = rand_idx[:, num_dst:, :] # src
        
        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst
            
        # Compute similarity scores
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        
        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)
        
        # Find most similar pairs greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        return torch.cat([unm, dst], dim=1)
        
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape
        
        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        
        # Combine back to original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)
        
        return out
        
    return merge, unmerge