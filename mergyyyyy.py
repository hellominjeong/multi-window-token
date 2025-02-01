# import torch
# from typing import Tuple, Callable
# from torch.nn import functional as F


# def do_nothing(x: torch.Tensor, mode:str=None):
#     return x


# def mps_gather_workaround(input, dim, index):
#     if input.shape[-1] == 1:
#         return torch.gather(
#             input.unsqueeze(-1),
#             dim - 1 if dim < 0 else dim,
#             index.unsqueeze(-1)
#         ).squeeze(-1)
#     else:
#         return torch.gather(input, dim, index)


# def create_window_partition_indices(h: int, w: int, window_size: int, device: torch.device) -> torch.Tensor:
#     """Create indices for window partitioning.
#     Assumes h and w are divisible by window_size (e.g., 64x64 for SD)"""
#     indices = torch.arange(h * w, device=device).reshape(h, w)
#     num_h = h // window_size
#     num_w = w // window_size
    
#     # Partition windows
#     indices = indices.reshape(num_h, window_size, num_w, window_size)
#     indices = indices.permute(0, 2, 1, 3).reshape(-1, window_size * window_size)
    
#     return indices


# def merge_similar_windows(tokens: torch.Tensor, indices: torch.Tensor, 
#                          window_size: int, merge_ratio: float,
#                          h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Merge similar windows by comparing entire window contents
    
#     Args:
#         tokens: [B, N, C] token features
#         indices: [(h*w)/(window_size*window_size), window_size*window_size] window partition indices
#         window_size: size of each window
#         merge_ratio: ratio of windows to merge
#         h, w: original height and width
#     """
#     B, N, C = tokens.shape
#     num_windows = indices.shape[0]
#     tokens_per_window = window_size * window_size
    
#     # Reshape tokens into windows [B, num_windows, window_size*window_size, C]
#     window_tokens = tokens.gather(1, indices.view(1, -1, 1).expand(B, -1, C))
#     window_tokens = window_tokens.view(B, num_windows, tokens_per_window, C)
    
#     # Normalize each window's tokens
#     window_tokens = window_tokens / (window_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    
#     # Calculate full similarity matrices between windows 
#     # [B, num_windows, num_windows, tokens_per_window, tokens_per_window]
#     window_similarity = torch.zeros(B, num_windows, num_windows, 
#                                   tokens_per_window, tokens_per_window, 
#                                   device=tokens.device)
    
#     # Calculate token-wise similarity matrix for each pair of windows
#     for i in range(num_windows):
#         for j in range(i+1, num_windows):
#             # Full token-wise similarity matrix [B, tokens_per_window, tokens_per_window]
#             token_similarity = torch.matmul(
#                 window_tokens[:, i],  # [B, tokens_per_window, C]
#                 window_tokens[:, j].transpose(-2, -1)  # [B, C, tokens_per_window]
#             )
            
#             # Store full similarity matrices
#             window_similarity[:, i, j] = token_similarity
#             window_similarity[:, j, i] = token_similarity.transpose(-2, -1)
    
#     # Find most similar window pairs based on the maximum token similarities
#     # Compute average similarity between windows
#     sim_scores = torch.matmul(
#         window_tokens.view(B, num_windows, -1),  # [B, num_windows, tokens_per_window*C]
#         window_tokens.view(B, num_windows, -1).transpose(-2, -1)  # [B, num_windows, tokens_per_window*C]
#     )  # [B, num_windows, num_windows]
    
#     # Mask self-similarity
#     mask = torch.eye(num_windows, device=tokens.device).bool()
#     sim_scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
    
#     # Number of window pairs to merge
#     windows_to_merge = int(num_windows * merge_ratio)
    
#     # Find most similar window pairs
#     values, pairs = sim_scores.reshape(B, -1).topk(windows_to_merge, dim=-1)
#     src_windows = pairs // num_windows
#     dst_windows = pairs % num_windows
    
#     # Merge similar windows
#     merged_windows = []
#     merge_indices = []
    
#     for b in range(B):
#         current_windows = window_tokens[b].clone()  # [num_windows, tokens_per_window, C]
#         window_pairs = []
        
#         # Process each pair of similar windows
#         for src_win, dst_win in zip(src_windows[b], dst_windows[b]):
#             src_window = current_windows[src_win]  # [tokens_per_window, C]
#             dst_window = current_windows[dst_win]  # [tokens_per_window, C]
            
#             # Get token-level similarity between these windows
#             token_sim = window_similarity[b, src_win, dst_win]  # [tokens_per_window, tokens_per_window]
            
#             # Find best token matches
#             matched_indices = token_sim.max(dim=-1)[1]  # [tokens_per_window]
            
#             # Merge tokens based on their matched pairs
#             merged_window = torch.zeros_like(src_window)
#             for i, j in enumerate(matched_indices):
#                 merged_window[i] = (src_window[i] + dst_window[j]) / 2
                
#             # Update both windows with merged result
#             current_windows[src_win] = merged_window
#             current_windows[dst_win] = merged_window[matched_indices]  # Align with best matches
            
#             window_pairs.append(torch.tensor([src_win, dst_win], device=tokens.device))
            
#         merged_windows.append(current_windows)
#         merge_indices.append(torch.stack(window_pairs))
    
#     merged_windows = torch.stack(merged_windows)  # [B, num_windows, tokens_per_window, C]
#     merge_indices = torch.stack(merge_indices)  # [B, num_pairs, 2]
    
#     # Reshape back to original format
#     merged_tokens = merged_windows.reshape(B, -1, C)
    
#     return merged_tokens, merge_indices


# # def multi_scale_merge(metric: torch.Tensor,
# #                      w: int, h: int,
# #                      window_sizes: list,
# #                      merge_ratios: list,
# #                      no_rand: bool = False,
# #                      generator: torch.Generator = None) -> Tuple[Callable, Callable]:
# #     """
# #     Performs multi-scale token merging using different window sizes.
# #     Each scale is processed independently.
    
# #     Args:
# #      - metric [B, N, C]: metric to use for similarity
# #      - w: image width in tokens
# #      - h: image height in tokens
# #      - window_sizes: list of window sizes to use (e.g., [4, 16])
# #      - merge_ratios: list of merge ratios for each window size
# #      - no_rand: if true, disable randomness (not used in current implementation)
# #      - generator: random number generator (not used in current implementation)
# #     """
# #     B, N, C = metric.shape
# #     device = metric.device
    
# #     # Cache key based on tensor shape and parameters
# #     cache_key = f"{metric.shape}_{window_sizes}_{merge_ratios}"
    
# #     # Use cached results if available
# #     if hasattr(multi_scale_merge, 'cache') and cache_key in multi_scale_merge.cache:
# #         scale_indices = multi_scale_merge.cache[cache_key]
# #     else:
# #         # Initialize cache
# #         if not hasattr(multi_scale_merge, 'cache'):
# #             multi_scale_merge.cache = {}

# #         with torch.no_grad():
# #             # Normalize metrics for cosine similarity
# #             metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
            
# #             # Process each scale independently
# #             scale_tokens = []
# #             scale_indices = []
            
# #             for window_size, merge_ratio in zip(window_sizes, merge_ratios):
# #                 # Create window partition indices
# #                 indices = create_window_partition_indices(h, w, window_size, device)
                
# #                 # Merge similar windows at current scale
# #                 merged_tokens, merge_idx = merge_similar_windows(
# #                     metric, indices, window_size, merge_ratio, h, w)
# #                 print(f"[DEBUG] merged_tokens.shape: {merged_tokens.shape}, window_size: {window_size}")
# #                 print(f"[DEBUG] merge_idx.shape: {merge_idx.shape}")
# #                 # Calculate new dimensions
# #                 print(f"[DEBUG] Input size: (h={h}, w={w}), Window size: {window_size}")



# #                 n_h = h // window_size
# #                 n_w = w // window_size
# #                 print(f"[DEBUG] Calculated window dimensions: (n_h={n_h}, n_w={n_w})")
# #                 n_tokens = n_h * n_w * (window_size * window_size)
                
# #                 # Reshape merged tokens back to 2D spatial layout
# #                 # First ensure the total number of tokens is correct
# #                 merged_tokens = merged_tokens[:, :n_tokens, :]
# #                 tokens_2d = merged_tokens.view(B, n_h, n_w, window_size, window_size, C)
# #                 tokens_2d = tokens_2d.permute(0, 1, 3, 2, 4, 5).reshape(B, h, w, C)
                
# #                 # Interpolate to original size if needed
# #                 if window_size > 1:
# #                     tokens_2d = F.interpolate(
# #                         tokens_2d.permute(0, 3, 1, 2),
# #                         size=(h, w),
# #                         mode='bilinear',
# #                         align_corners=False
# #                     ).permute(0, 2, 3, 1)
                
# #                 scale_tokens.append(tokens_2d)
# #                 scale_indices.append((indices, merge_idx))
# #                 multi_scale_merge.cache[cache_key] = scale_indices
        
# #         def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
# #             """Merge tokens using pre-computed indices for each scale"""
# #             print("[DEBUG] Entering merge function")
# #             results = []
# #             for window_size, (indices, merge_idx) in zip(window_sizes, scale_indices):
# #                 # Apply merging at current scale
# #                 merged_tokens, _ = merge_similar_windows(
# #                     x, indices, window_size, merge_ratio, h, w)
                
# #                 # Calculate new dimensions
# #                 n_h = h // window_size
# #                 n_w = w // window_size
# #                 n_tokens = n_h * n_w * (window_size * window_size)
                
# #                 # Reshape and interpolate
# #                 merged_tokens = merged_tokens[:, :n_tokens, :]
# #                 tokens_2d = merged_tokens.view(B, n_h, n_w, window_size, window_size, C)
# #                 tokens_2d = tokens_2d.permute(0, 1, 3, 2, 4, 5).reshape(B, h, w, C)
                
# #                 if window_size > 1:
# #                     tokens_2d = F.interpolate(
# #                         tokens_2d.permute(0, 3, 1, 2),
# #                         size=(h, w),
# #                         mode='bilinear',
# #                         align_corners=False
# #                     ).permute(0, 2, 3, 1)
                
# #                 results.append(tokens_2d)
            
# #             # Average results from all scales
# #             return torch.stack(results).mean(dim=0).reshape(B, -1, C)

# #         def unmerge(x: torch.Tensor) -> torch.Tensor:
# #             """Unmerge tokens back to original positions"""
# #             print("[DEBUG] Entering unmerge function")
# #             results = []
# #             for window_size, (indices, merge_idx) in zip(window_sizes, scale_indices):
# #                 tokens_2d = x.view(B, h, w, C)
                
# #                 # Downsample if needed
# #                 if window_size > 1:
# #                     tokens_2d = F.interpolate(
# #                         tokens_2d.permute(0, 3, 1, 2),
# #                         size=(h // window_size, w // window_size),
# #                         mode='bilinear',
# #                         align_corners=False
# #                     ).permute(0, 2, 3, 1)
                
# #                 # Reshape to match window structure
# #                 n_h = h // window_size
# #                 n_w = w // window_size
# #                 tokens_per_window = window_size * window_size

# #                 # 1) 먼저 tokens_2d를 window 구조로 재구성
# #                 reshaped_tokens = tokens_2d.reshape(B, n_h, n_w, -1)  # [B, n_h, n_w, C]
                
# #                 # 2) window 단위로 처리할 수 있도록 변환
# #                 window_tokens = reshaped_tokens.reshape(B, n_h * n_w, -1)  # [B, num_windows, C]

# #                 # Apply unmerging using saved indices
# #                 unmerged_tokens = []
# #                 for b in range(B):
# #                     current = window_tokens[b]  # [num_windows, C]
# #                     src_wins = merge_idx[b, :, 0]
# #                     dst_wins = merge_idx[b, :, 1]
                    
# #                     # Restore original values
# #                     unmerged = current.clone()
# #                     unmerged[src_wins] = current[dst_wins]
# #                     unmerged_tokens.append(unmerged)
                
# #                 unmerged = torch.stack(unmerged_tokens)  # [B, num_windows, C]
                
# #                 # Reshape back to spatial layout
# #                 unmerged = unmerged.reshape(B, n_h, n_w, -1)
                
# #                 # Interpolate back to original size if needed
# #                 if window_size > 1:
# #                     unmerged = F.interpolate(
# #                         unmerged.permute(0, 3, 1, 2),
# #                         size=(h, w),
# #                         mode='bilinear',
# #                         align_corners=False
# #                     ).permute(0, 2, 3, 1)
                
# #                 results.append(unmerged.reshape(B, -1, C))
            
# #             # Average results from all scales
# #             return torch.stack(results).mean(dim=0)


# #     return merge, unmerge
# def multi_scale_merge(metric: torch.Tensor, w: int, h: int, window_sizes: list, merge_ratios: list, no_rand: bool = False, generator: torch.Generator = None) -> Tuple[Callable, Callable]:
#     B, N, C = metric.shape
#     device = metric.device
    
#     if len(window_sizes) != len(merge_ratios):
#         raise ValueError("window_sizes and merge_ratios must have same length")
    
#     # 초기 처리 한 번만 수행
#     with torch.no_grad():
#         metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
#         merged_results = {}
        
#         for window_size, merge_ratio in zip(window_sizes, merge_ratios):
#             indices = create_window_partition_indices(h, w, window_size, device)
#             merged_tokens, merge_idx = merge_similar_windows(
#                 metric, indices, window_size, merge_ratio, h, w)
            
#             # 공간 구조로 변환하고 원래 크기로 보간
#             n_h = h // window_size
#             n_w = w // window_size
#             tokens_2d = merged_tokens.view(B, n_h * window_size, n_w * window_size, C)
            
#             if window_size > 1:
#                 tokens_2d = F.interpolate(
#                     tokens_2d.permute(0, 3, 1, 2),
#                     size=(h, w),
#                     mode='bilinear',
#                     align_corners=False
#                 ).permute(0, 2, 3, 1)
            
#             # 모든 윈도우 크기에 대해 동일한 크기의 출력 유지
#             tokens_flat = tokens_2d.reshape(B, h * w, C)
            
#             # 결과 저장
#             merged_results[window_size] = {
#                 'tokens': tokens_flat,
#                 'indices': indices,
#                 'merge_idx': merge_idx
#             }
            
#             print(f"[DEBUG] Window size {window_size}: token shape {tokens_flat.shape}")
    
#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         results = []
        
#         for window_size in window_sizes:
#             saved = merged_results[window_size]
#             # 이미 올바른 크기로 저장되어 있음
#             results.append(saved['tokens'])
        
#         # 모든 결과가 같은 크기이므로 안전하게 스택 가능
#         stacked = torch.stack(results)  # [num_scales, B, N, C]
#         return stacked.mean(dim=0)  # [B, N, C]

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         return x  # 이미 올바른 크기로 저장되어 있으므로 그대로 반환

#     print(f"[DEBUG] Initial processing complete. Results stored for window sizes: {window_sizes}")
#     return merge, unmerge

import torch
from typing import Callable, Tuple

def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

# def multi_scale_soft_matching(
#     input_tensor: torch.Tensor, 
#     w: int, 
#     h: int, 
#     scales: Tuple[int], 
#     r: int, 
#     no_rand: bool = False,
#     generator: torch.Generator = None
# ) -> Tuple[Callable, Callable]:
#     """
#     Multi-scale token merging and unmerging with similarity calculation.

#     Args:
#         input_tensor (torch.Tensor): Input tensor [B, N, C]
#         w (int): Image width in tokens
#         h (int): Image height in tokens
#         scales (Tuple[int]): Tuple of scales for partitioning (e.g., (3, 16))
#         r (int): Number of tokens to merge
#         no_rand (bool): Disable randomness for partitioning
#         generator (torch.Generator): Random number generator

#     Returns:
#         Tuple[Callable, Callable]: Merge and unmerge functions
#     """
#     B, N, C = input_tensor.shape

#     if r <= 0:
#         return do_nothing, do_nothing

#     gather = mps_gather_workaround if input_tensor.device.type == "mps" else torch.gather

#     with torch.no_grad():
#         merged_indices = []
#         for scale in scales:
#             sx, sy = scale, scale
#             hsy, wsx = h // sy, w // sx

#             if no_rand:
#                 rand_idx = torch.zeros(hsy, wsx, 1, device=input_tensor.device, dtype=torch.int64)
#             else:
#                 rand_idx = torch.randint(
#                     sy * sx, size=(hsy, wsx, 1), device=input_tensor.device, generator=generator
#                 )

#             idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=input_tensor.device, dtype=torch.int64)
#             idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
#             idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

#             if (hsy * sy) < h or (wsx * sx) < w:
#                 idx_buffer = torch.zeros(h, w, device=input_tensor.device, dtype=torch.int64)
#                 idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
#             else:
#                 idx_buffer = idx_buffer_view

#             rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
#             num_dst = hsy * wsx
#             a_idx = rand_idx[:, num_dst:, :]
#             b_idx = rand_idx[:, :num_dst, :]

#             metric = input_tensor / input_tensor.norm(dim=-1, keepdim=True)
#             src = gather(metric, dim=1, index=a_idx.expand(B, a_idx.shape[1], C))
#             dst = gather(metric, dim=1, index=b_idx.expand(B, b_idx.shape[1], C))

#             scores = src @ dst.transpose(-1, -2)
#             num_merge = min(src.shape[1], r // len(scales))

#             node_max, node_idx = scores.max(dim=-1)
#             edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
#             merged_indices.append((a_idx, b_idx, edge_idx, node_idx, num_merge))

#         def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#             merged = []
#             for scale_idx, (a_idx, b_idx, edge_idx, node_idx, num_merge) in enumerate(merged_indices):
#                 unm_idx = edge_idx[..., num_merge:, :]
#                 src_idx = edge_idx[..., :num_merge, :]
#                 dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

#                 src = gather(x, dim=1, index=a_idx.expand(B, a_idx.shape[1], C))
#                 dst = gather(x, dim=1, index=b_idx.expand(B, b_idx.shape[1], C))

#                 dst = dst.scatter_reduce(-2, dst_idx.expand(B, num_merge, C), src, reduce=mode)
#                 merged.append(dst)

#             return torch.cat(merged, dim=1)

#         def unmerge(x: torch.Tensor) -> torch.Tensor:
#             restored = torch.zeros_like(input_tensor)
#             offset = 0
#             for scale_idx, (a_idx, b_idx, edge_idx, node_idx, num_merge) in enumerate(merged_indices):
#                 num_dst = b_idx.shape[1]
#                 dst = x[:, offset:offset + num_dst, :]
#                 restored.scatter_(-2, b_idx.expand(B, num_dst, C), dst)
#                 offset += num_dst

#             return restored

#         return merge, unmerge
#-----------
# import torch
# from typing import Callable, Tuple


# def multi_scale_soft_matching(
#     input_tensor: torch.Tensor,
#     w: int,
#     h: int,
#     scales: Tuple[int],
#     r: int,
#     no_rand: bool = False,
#     generator: torch.Generator = None
# ) -> Tuple[Callable, Callable]:
#     """
#     Multi-scale window-based token merging with similarity calculation.

#     Args:
#         input_tensor (torch.Tensor): Input tensor [B, N, C]
#         w (int): Image width in tokens
#         h (int): Image height in tokens
#         scales (Tuple[int]): Tuple of scales for window sizes (e.g., (3, 16))
#         r (int): Number of windows to merge
#         no_rand (bool): Disable randomness for window selection
#         generator (torch.Generator): Random number generator

#     Returns:
#         Tuple[Callable, Callable]: Merge and unmerge functions
#     """
#     B, N, C = input_tensor.shape
    
#     if r <= 0:
#         return lambda x: x, lambda x: x

#     with torch.no_grad():
#         merged_indices = []
#         for scale in scales:
#             # 윈도우 생성 과정
#             window_size = scale
#             num_windows_h = h // window_size
#             num_windows_w = w // window_size
#             num_windows = num_windows_h * num_windows_w
            
#             pad_h = (window_size - h % window_size) % window_size
#             pad_w = (window_size - w % window_size) % window_size
#             if pad_h > 0 or pad_w > 0:
#                 padded = torch.nn.functional.pad(
#                     input_tensor.view(B, h, w, C),
#                     (0, 0, 0, pad_w, 0, pad_h)
#                 )
#                 input_reshaped = padded.view(B, h + pad_h, w + pad_w, C)
#             else:
#                 input_reshaped = input_tensor.view(B, h, w, C)
            
#             # 윈도우 펼치기
#             windows = input_reshaped.unfold(1, window_size, window_size).\
#                 unfold(2, window_size, window_size)
#             windows = windows.contiguous().view(
#                 B, num_windows_h, num_windows_w, window_size, window_size, C
#             )
#             windows = windows.view(B, num_windows, window_size * window_size, C)  # [B, num_windows, W*W, C]

#             # 윈도우 간 유사도 계산
#             similarity_matrix = torch.zeros(B, num_windows, num_windows, device=input_tensor.device)
#             for batch_idx in range(B):
#                 for i in range(num_windows):
#                     for j in range(num_windows):
#                         # 윈도우 a와 b의 모든 원소 간 코사인 유사도 계산
#                         a = windows[batch_idx, i]  # [W*W, C]
#                         b = windows[batch_idx, j]  # [W*W, C]
                        
#                         # 코사인 유사도 계산
#                         a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
#                         b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
#                         pairwise_similarity = torch.matmul(a_norm, b_norm.T)  # [W*W, W*W]
                        
#                         # 평균 유사도 계산
#                         # b = b.item() if isinstance(b, torch.Tensor) else b
#                         # i = i.item() if isinstance(i, torch.Tensor) else i
#                         # j = j.item() if isinstance(j, torch.Tensor) else j
#                         similarity_matrix[batch_idx, i, j] = pairwise_similarity.mean()

#             # 유사도 행렬에서 병합 쌍 선택 (기존과 동일)
#             mask = torch.eye(num_windows, device=similarity_matrix.device).bool().unsqueeze(0).expand(B, -1, -1) #자기자신 유사도 제외
#             similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
            
#             values, nn_indices = similarity_matrix.max(dim=-1)
#             requested_merges = r // len(scales)
#             target_merges = min(num_windows // 2, requested_merges)
#             num_merge = min(target_merges, values.size(1))
            
#             top_values, top_indices = values.topk(num_merge, dim=-1)
#             merge_indices = torch.stack([top_indices, nn_indices.gather(1, top_indices)], dim=-1)
            
#             window_info = {
#                 'window_size': window_size,
#                 'num_windows_h': num_windows_h,
#                 'num_windows_w': num_windows_w,
#                 'num_windows': num_windows,
#                 'pad_h': pad_h,
#                 'pad_w': pad_w,
#                 'merge_indices': merge_indices,
#                 'num_merge': num_merge
#             }
#             merged_indices.append(window_info)
            
#         def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#             merged_results = []
            
#             for scale_info in merged_indices:
#                 window_size = scale_info['window_size']
#                 num_windows_h = scale_info['num_windows_h']
#                 num_windows_w = scale_info['num_windows_w']
#                 num_windows = scale_info['num_windows']
#                 pad_h = scale_info['pad_h']
#                 pad_w = scale_info['pad_w']
#                 indices = scale_info['merge_indices']
#                 num_merge = scale_info['num_merge']
                
#                 # Reshape input into windows
#                 if pad_h > 0 or pad_w > 0:
#                     padded = torch.nn.functional.pad(
#                         x.view(B, h, w, C),
#                         (0, 0, 0, pad_w, 0, pad_h)
#                     )
#                     x_reshaped = padded.view(B, h + pad_h, w + pad_w, C)
#                 else:
#                     x_reshaped = x.view(B, h, w, C)
                
#                 windows = x_reshaped.unfold(1, window_size, window_size).\
#                     unfold(2, window_size, window_size)
#                 windows = windows.contiguous().view(
#                     B, num_windows, window_size * window_size, C
#                 )
                
#                 # Prepare output tensor (with same dtype as input)
#                 available_windows = windows.size(1)  # 실제 윈도우 수
#                 num_output_windows = available_windows - num_merge
                
#                 merged = torch.zeros(
#                     B, num_output_windows, window_size * window_size, C,
#                     device=x.device, dtype=x.dtype
#                 )
                
#                 # First copy all unmmerged windows
#                 merged[:, :available_windows-2*num_merge] = windows[:, 2*num_merge:]
                
#                 # Then handle merged pairs
#                 # 벡터화된 병합 수행------------------------------------------에러에러에ㅓ레ㅓ
#                 for b in range(B):
#                     merge_idx = available_windows - 2*num_merge  #///////////
#                     num_merge = merge_indices.size(1) #////////////
#                     if num_merge > 0:
#                         idx1 = merge_indices[b, :, 0]
#                         idx2 = merge_indices[b, :, 1]

#                         if torch.max(idx1).item() >= windows.size(1) or torch.max(idx2).item() >= windows.size(1):
#                             raise ValueError(f"Index out of bounds for windows: idx1={idx1}, idx2={idx2}, windows size={windows.size(1)}")
#                         print(f"merge_indices shape: {merge_indices.shape}")
#                         print(f"merge_indices[b, :, 0]: {merge_indices[b, :, 0]}")
#                         print(f"merge_indices[b, :, 1]: {merge_indices[b, :, 1]}")
#                         merged_pairs = (windows[b, idx1] + windows[b, idx2]) / 2 if mode == "mean" else windows[b, idx1] + windows[b, idx2]
#                         expected_size = merged[b, merge_idx:merge_idx + num_merge].shape
#                         if merged_pairs.shape != expected_size:
#                             raise ValueError(
#                                 f"Size mismatch: merged_pairs={merged_pairs.shape}, expected={expected_size}"
#                             )
#                         merged[b, merge_idx:merge_idx + num_merge] = merged_pairs

                
#                 merged_results.append(merged.view(B, -1, C))
            
#             return torch.cat(merged_results, dim=1)
        
#         def unmerge(x: torch.Tensor) -> torch.Tensor:
#             unmerged = torch.zeros_like(input_tensor)
#             offset = 0
            
#             for scale_info in merged_indices:
#                 window_size = scale_info['window_size']
#                 num_windows_h = scale_info['num_windows_h']
#                 num_windows_w = scale_info['num_windows_w']
#                 num_windows = scale_info['num_windows']
#                 indices = scale_info['merge_indices']
#                 num_merge = scale_info['num_merge']
                
#                 # Extract windows for this scale
#                 num_retained = num_windows - num_merge
#                 scale_windows = x[:, offset:offset + num_retained * window_size * window_size, :].view(
#                     B, num_retained, window_size * window_size, C
#                 )
                
#                 # Reshape to match original window layout
#                 unmerged_windows = torch.zeros(
#                     B, num_windows, window_size * window_size, C,
#                     device=x.device, dtype=x.dtype
#                 )
                
#                 # First restore the unmerged windows
#                 unmerged_windows[:, 2*num_merge:] = scale_windows[:, :num_windows-2*num_merge]
                
#                 # Then restore merged pairs
#                 for b in range(B):
#                     merged_content = scale_windows[b, num_windows-2*num_merge:]
#                     # Get actual number of merged pairs for this scale
#                     actual_merges = min(num_merge, merge_indices.size(1))
#                     for i in range(actual_merges):
#                         idx1, idx2 = merge_indices[b, i]
#                         if i < merged_content.size(0):  # Check if we have content for this index
#                             unmerged_windows[b, idx1] = merged_content[i]
#                             unmerged_windows[b, idx2] = merged_content[i]
                
#                 # Reshape windows back to original format
#                 unmerged_windows = unmerged_windows.view(
#                     B, num_windows_h, num_windows_w, window_size, window_size, C
#                 )
#                 unmerged_windows = unmerged_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
#                 unmerged_scale = unmerged_windows.view(B, -1, C)
                
#                 # Add to final result
#                 window_tokens = window_size * window_size
#                 scale_tokens = num_windows * window_tokens
#                 unmerged[:, :scale_tokens, :] += unmerged_scale[:, :scale_tokens, :]
                
#                 offset += num_retained * window_size * window_size
            
#             return unmerged
        
#         return merge, unmerge

import torch
from typing import Tuple, Callable
import torch.nn.functional as F
def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def create_windows(metric: torch.Tensor, w: int, h: int, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Converts token data to windows.

    Args:
        metric (torch.Tensor): Input tensor [B, N, C].
        w (int): Width of the image in tokens.
        h (int): Height of the image in tokens.
        window_size (int): Size of the window.

    Returns:
        Tuple[torch.Tensor, int, int]: Windows tensor, num_windows_h, num_windows_w.
    """
    # print("create_windows----")
    B, N, C = metric.shape
    assert N == w * h, "The metric size doesn't match width and height."

    # Reshape to [B, H, W, C]
    metric_reshaped = metric.view(B, h, w, C)

    # Create windows: [B, num_windows_h, num_windows_w, window_size, window_size, C]
    num_windows_h = h // window_size
    num_windows_w = w // window_size
    windows = metric_reshaped.unfold(1, window_size, window_size).unfold(2, window_size, window_size)

    # Flatten windows: [B, num_windows, window_size * window_size, C]
    windows = windows.contiguous().view(
        B, num_windows_h * num_windows_w, window_size * window_size, C
    )
    
    return windows, num_windows_h, num_windows_w

def calculate_similarity(windows: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity between windows without averaging.

    Args:
        windows (torch.Tensor): Windows tensor [B, num_windows, W*W, C].

    Returns:
        torch.Tensor: Similarity matrix [B, num_windows, num_windows].
    """
    # [B, num_windows, W*W, C] -> [B, num_windows, W*W*C]
    window_features = windows.flatten(start_dim=2)  # 윈도우 내부 모든 토큰을 하나의 벡터로 변환

    # 정규화된 특징을 한 번에 계산
    norm = torch.norm(window_features, dim=-1, keepdim=True)
    normalized_features = window_features / (norm + 1e-6)

    # 배치 행렬 곱 사용하여 코사인 유사도 계산
    return torch.bmm(normalized_features, normalized_features.transpose(-2, -1))

def merge_windows(windows: torch.Tensor, similarity_matrix: torch.Tensor, r: int, mode="mean") -> torch.Tensor:
    """
    Merge similar windows using similarity matrix and remove r windows.

    Args:
        windows (torch.Tensor): Windows tensor [B, num_windows, window_size, C].
        similarity_matrix (torch.Tensor): Similarity matrix [B, num_windows, num_windows].
        r (int): Number of merges to perform.
        mode (str): Merge mode ('mean' or 'sum').

    Returns:
        torch.Tensor: Reduced merged windows tensor [B, num_windows - r, window_size, C].
    """
    # print("merge_windows----")
    B, num_windows, window_size, C = windows.shape
    # similarity_matrix = similarity_matrix.clone()
    # similarity_matrix[:, torch.arange(num_windows), torch.arange(num_windows)] = float('-inf')
    # # ✅ Merge 수행 (r개의 윈도우를 선택)
    # valid_r = min(r, num_windows // 2, similarity_matrix.shape[-1]) 
    # values, indices = similarity_matrix.view(B, -1).topk(valid_r, dim=-1)  

    # idx1 = indices // num_windows  # 첫 번째 윈도우 인덱스
    # idx2 = indices % num_windows  # 두 번째 윈도우 인덱스
    
    similarity_matrix = similarity_matrix.clone()
    similarity_matrix[:, torch.arange(num_windows), torch.arange(num_windows)] = float('-inf')  # 자기 자신 제외
    r_windows = int(num_windows * 0.5)  # 입력 텐서 기반이 아니라, 윈도우 개수 기준으로 변환
    valid_r = min(r_windows, num_windows // 2)  # 동일한 기준으로 비교
    # valid_r = min(r, num_windows // 2, similarity_matrix.shape[-1]) 
    # print("valid_r: ",valid_r)
    # print("r: ",r_windows)
    # ✅ 선택된 윈도우를 담을 리스트
    selected_idx1 = []
    selected_idx2 = []
    num_selected = 0
    # print(f"num_selected: {num_selected}, valid_r: {valid_r}")
    while num_selected < valid_r:
        # print("in while ---- sim matrix before view", similarity_matrix.shape)
        values, indices = similarity_matrix.view(B, -1).topk(valid_r - num_selected, dim=-1)  # 아직 필요한 개수만큼만 가져오기
        new_idx1 = indices // num_windows
        new_idx2 = indices % num_windows


        valid_pairs = new_idx1 != new_idx2 
        # print("valid_pairs",valid_pairs.shape)
        valid_pairs = valid_pairs.view(B, -1)

        # new_idx2 = new_idx2[valid_pairs]
        new_idx1 = torch.where(valid_pairs, new_idx1, -1)  # -1을 채워서 구조 유지
        new_idx2 = torch.where(valid_pairs, new_idx2, -1)

        valid_range = (new_idx1 < num_windows) & (new_idx2 < num_windows)
        # print("valid_range",valid_range.shape)
        valid_range = valid_range.view(B, -1)  # 배치 차원 유

        new_idx1 = torch.where(valid_range, new_idx1, -1)
          # -1을 채워서 구조 유지
        new_idx2 = torch.where(valid_range, new_idx2, -1)


        # # ✅ 3. `num_selected` 업데이트
        # print("new_idx1.shape", new_idx1.shape)
        num_selected += new_idx1.shape[1]  # 실제 병합된 개수만큼 업데이트
        # print(f"Updated num_selected: {num_selected}/{valid_r}")

        # ✅ 만약 num_selected가 더 이상 증가하지 않으면 (예: 모든 병합 가능한 윈도우를 다 찾았을 때) 종료
        if new_idx1.shape[0] == 0:
            print("⚠ No more valid pairs to merge. Exiting loop.")
            break
 
    # # ✅ 병합 수행
    merged = (windows[:, new_idx1] + windows[:, new_idx2]) / 2  
    merged = merged.mean(dim=1)  # ✅ 차원 조정
  
    actual_r = new_idx1.shape[1]  # 실제 병합된 개수 반영

    if actual_r == 0:
        print("⚠ Warning: No valid merge pairs found. Skipping merge step.")
        return windows
 
    merged_windows = merged
    # print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

    return merged_windows, actual_r


# def window_based_soft_matching(
#     input_tensor: torch.Tensor,
#     w: int,
#     h: int,
#     r: int,
#     mode="mean"
# ) -> torch.Tensor:
#     """
#     Perform window-based soft matching and merging.

#     Args:
#         input_tensor (torch.Tensor): Input tensor [B, N, C].
#         w (int): Width of the image in tokens.
#         h (int): Height of the image in tokens.
#         r (int): Number of merges to perform.
#         mode (str): Merge mode ('mean' or 'sum').

#     Returns:
#         torch.Tensor: Reconstructed tensor [B, N, C].
#     """
#     B, N, C = input_tensor.shape
#     device = input_tensor.device
#     # print("input tensor: ===========================")
#     # print(input_tensor.shape)
#     windows_4, num_windows_h_4, num_windows_w_4 = create_windows(input_tensor, w, h, window_size=4)
#     similarity_matrix_4 = calculate_similarity(windows_4)
#     merged_windows_4, numw4 = merge_windows(windows_4, similarity_matrix_4, r, mode)
    
#     windows_8, num_windows_h_8, num_windows_w_8 = create_windows(input_tensor, w, h, window_size=8)
#     similarity_matrix_8 = calculate_similarity(windows_8)
#     merged_windows_8, numw8 = merge_windows(windows_8, similarity_matrix_8, r, mode)

#     windows_16, num_windows_h_16, num_windows_w_16 = create_windows(input_tensor, w, h, window_size=16)
#     similarity_matrix_16 = calculate_similarity(windows_16)
#     merged_windows_16, numw16 = merge_windows(windows_16, similarity_matrix_16, r, mode)

#     merged_windows_4 = merged_windows_4.permute(0, 3, 1, 2)
#     merged_windows_16 = merged_windows_16.permute(0, 3, 1, 2)
#     target_size = (merged_windows_8.shape[1], merged_windows_8.shape[2])  # (102, 16)
#     print("---------------------")
#     print("merged_windows_4: ",merged_windows_4.shape)
#     print("merged_windows_8: ",merged_windows_8.shape)
#     print("merged_windows_16: ",merged_windows_16.shape)

#     print("------------------")
#     merged_windows_4_resized = F.interpolate(merged_windows_4, 
#                                         size=target_size,  # (num_windows, height)
#                                         mode='bilinear', align_corners=False)  # Bilinear Interpolation 사용
    
#     merged_windows_16_resized = F.interpolate(merged_windows_16, 
#                                         size=target_size,  # (num_windows, height)
#                                         mode='bilinear', align_corners=False)  # Bilinear Interpolation 사용
    
#     merged_windows_4_final = merged_windows_4_resized.permute(0, 2, 3, 1)
#     merged_windows_16_final = merged_windows_16_resized.permute(0, 2, 3, 1)
#     print("merged_windows_4_final: ",merged_windows_4_final.shape)
#     print("merged_windows_8: ",merged_windows_8.shape)
#     print("merged_windows_16_final: ",merged_windows_16_final.shape)
#     print("----------------")
#     weights = torch.tensor([0.2, 0.5, 0.3], device = input_tensor.device).view(1,1,1,3)
#     merged_x = (weights[...,0] * merged_windows_4_final + 
#                 weights[...,1]  * merged_windows_8 + 
#                 weights[...,2] * merged_windows_16_final ) 
#     print(merged_x.shape)
    
#     def unmerge_fn(merged_x: torch.Tensor, h: int, w: int) -> torch.Tensor:
#         reconstructed = F.interpolate(merged_x.permute(0,3,1,2), 
#                                     size = (h,w), mode = 'bilinear', align_corners = False).permute(0,2,3,1)
#         return reconstructed

#     a = unmerge_fn(merged_x, h, w)
#     print("---------------", type(a))
#     return merged_x, a  

def window_based_soft_matching(input_tensor: torch.Tensor, w: int, h: int, r: int, mode="mean") -> torch.Tensor:

    B, N, C = input_tensor.shape
    device = input_tensor.device

    windows_4, num_windows_h_4, num_windows_w_4 = create_windows(input_tensor, w, h, window_size=4)
    similarity_matrix_4 = calculate_similarity(windows_4)
    merged_windows_4, numw4 = merge_windows(windows_4, similarity_matrix_4, r, mode)
    
    windows_8, num_windows_h_8, num_windows_w_8 = create_windows(input_tensor, w, h, window_size=8)
    similarity_matrix_8 = calculate_similarity(windows_8)
    merged_windows_8, numw8 = merge_windows(windows_8, similarity_matrix_8, r, mode)

    windows_16, num_windows_h_16, num_windows_w_16 = create_windows(input_tensor, w, h, window_size=16)
    similarity_matrix_16 = calculate_similarity(windows_16)
    merged_windows_16, numw16 = merge_windows(windows_16, similarity_matrix_16, r, mode)

    merged_windows_4 = merged_windows_4.permute(0, 3, 1, 2)
    merged_windows_16 = merged_windows_16.permute(0, 3, 1, 2)
    target_size = (merged_windows_8.shape[1], merged_windows_8.shape[2])  # (102, 16)
    print("---------------------")
    print("merged_windows_4: ",merged_windows_4.shape)
    print("merged_windows_8: ",merged_windows_8.shape)
    print("merged_windows_16: ",merged_windows_16.shape)

    print("------------------")
    merged_windows_4_resized = F.interpolate(merged_windows_4, 
                                        size=target_size,  # (num_windows, height)
                                        mode='bilinear', align_corners=False)  # Bilinear Interpolation 사용
    
    merged_windows_16_resized = F.interpolate(merged_windows_16, 
                                        size=target_size,  # (num_windows, height)
                                        mode='bilinear', align_corners=False)  # Bilinear Interpolation 사용
    
    merged_windows_4_final = merged_windows_4_resized.permute(0, 2, 3, 1)
    merged_windows_16_final = merged_windows_16_resized.permute(0, 2, 3, 1)
    print("merged_windows_4_final: ",merged_windows_4_final.shape)
    print("merged_windows_8: ",merged_windows_8.shape)
    print("merged_windows_16_final: ",merged_windows_16_final.shape)
    print("----------------")
    result = torch.cat([merged_windows_4_final,merged_windows_8, merged_windows_16_final], dim=-1)
    print(result.shape)

    # weights = torch.tensor([0.2, 0.5, 0.3], device = input_tensor.device).view(1,1,1,3)
    # merged_x = (weights[...,0] * merged_windows_4_final + 
    #             weights[...,1]  * merged_windows_8 + 
    #             weights[...,2] * merged_windows_16_final ) 

    # print(merged_x.shape)

    def merge_fn(x):
        return result
    
    # def unmerge_fn(merged_x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    #     """
    #     Unmerge function to revert back to the original tensor shape.
    #     """

    #     reconstructed = F.interpolate(merged_x.permute(0,3,1,2), 
    #                                 size = (h,w), mode = 'bilinear', align_corners = False).permute(0,2,3,1)
    #     return reconstructed

    return merge_fn, unmerg_fn
