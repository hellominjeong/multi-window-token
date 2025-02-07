
import torch
from typing import Tuple, Callable
import math 
import torch.nn.functional as F

def do_nothing(x: torch.Tensor, mode:str=None):
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

import torch
import torch.nn.functional as F
import math
from typing import Tuple

def create_windows(metric: torch.Tensor, w: int, h: int, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Converts token data to windows.
    """
    B, N, C = metric.shape
    assert N == w * h, "The metric size doesn't match width and height."

    metric_reshaped = metric.view(B, h, w, C)
    num_windows_h = h // window_size
    num_windows_w = w // window_size

    windows = metric_reshaped.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    windows = windows.contiguous().view(B, num_windows_h * num_windows_w, window_size * window_size, C)

    return windows, num_windows_h, num_windows_w

def calculate_similarity(windows: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity between windows.
    """
    window_features = windows.flatten(start_dim=2)  
    norm = torch.norm(window_features, dim=-1, keepdim=True)
    normalized_features = window_features / (norm + 1e-6)
    return torch.bmm(normalized_features, normalized_features.transpose(-2, -1))

def merge_windows(windows: torch.Tensor, similarity_matrix: torch.Tensor, r: int) -> Tuple[torch.Tensor, list]:
    """
    Merge similar windows using similarity matrix and return merge indices.
    """
    B, num_windows, window_size, C = windows.shape
    similarity_matrix = similarity_matrix.clone()
    similarity_matrix[:, torch.arange(num_windows), torch.arange(num_windows)] = float('-inf')  

    valid_r = min(r, num_windows // 2)
    selected_idx1 = []
    selected_idx2 = []

    for _ in range(valid_r):
        values, indices = similarity_matrix.view(B, -1).topk(1, dim=-1)  
        idx1 = indices // num_windows
        idx2 = indices % num_windows
        valid_pairs = idx1 != idx2
        idx1 = torch.where(valid_pairs, idx1, -1)
        idx2 = torch.where(valid_pairs, idx2, -1)

        if (idx1 == -1).all():
            break  

        selected_idx1.append(idx1)
        selected_idx2.append(idx2)
        similarity_matrix[:, idx1, :] = float('-inf')
        similarity_matrix[:, :, idx2] = float('-inf')

    if not selected_idx1:
        return windows, []

    idx1 = torch.cat(selected_idx1, dim=-1)
    idx2 = torch.cat(selected_idx2, dim=-1)

    merged = (windows[:, idx1] + windows[:, idx2]) / 2  
    merged = merged.mean(dim=1)  

    merge_indices = list(zip(idx1.tolist(), idx2.tolist()))
    
    return merged, merge_indices

import torch
import math
import torch.nn.functional as F

def unmerge_windows(merged_x: torch.Tensor, merge_indices: list, h: int, w: int) -> torch.Tensor:
    """
    병합된 토큰을 원래 크기로 복원하는 함수.
    
    Args:
        merged_x (torch.Tensor): 병합된 토큰 텐서 [B, N, C]
        merge_indices (list): 병합된 토큰의 인덱스 정보
        h (int): 원본 이미지 높이
        w (int): 원본 이미지 너비
        
    Returns:
        torch.Tensor: 원래 크기로 복원된 토큰 [B, N, C]
    """
    B, N, C = merged_x.shape  # ✅ 이제 N을 직접 사용
    original_h, original_w = h, w  # 원본 이미지 크기 유지

    # ✅ 1. 원래 크기로 복원하기 위해 보간 (Interpolation)
    reconstructed = F.interpolate(
        merged_x.permute(0, 2, 1),  # [B, C, N] 형태로 변환
        size=(original_h * original_w),  # 원래 토큰 개수로 변환
        mode='linear',  # 선형 보간
        align_corners=False
    ).permute(0, 2, 1)  # 다시 [B, N, C] 형태로 변경

    # ✅ 2. merge_indices 기반 원본 위치 복원
    with torch.no_grad():
        for batch_idx in range(B):
            for idx1, idx2 in merge_indices:
                # ✅ 병합된 두 개의 토큰을 원래 위치로 복사 (mean 방식 적용)
                merged_token = reconstructed[batch_idx, idx1, :]  # 병합된 값 가져오기
                reconstructed[batch_idx, idx1, :] = merged_token  # 원래 위치로 복사
                reconstructed[batch_idx, idx2, :] = merged_token  # 쌍을 맞춘 위치에도 복사

    return reconstructed

    # B, num_windows, window_size, C = merged_x.shape
    # num_windows_h = h // int(math.sqrt(window_size))
    # num_windows_w = w // int(math.sqrt(window_size))

    # reconstructed = merged_x.view(B, num_windows_h, num_windows_w, 2, 2, C)

    # with torch.no_grad():
    #     for batch_idx in range(B):
    #         for idx1, idx2 in merge_indices:
    #             src_h, src_w = idx1 // 2, idx1 % 2
    #             dst_h, dst_w = idx2 // 2, idx2 % 2

    #             merged_token = reconstructed[batch_idx, :, :, src_h, src_w, :]
    #             reconstructed[batch_idx, :, :, src_h, src_w, :] = merged_token
    #             reconstructed[batch_idx, :, :, dst_h, dst_w, :] = merged_token

    # return reconstructed.view(B, h * w, C)

def window_based_soft_matching(input_tensor: torch.Tensor, w: int, h: int, r: int) -> Tuple[Callable, Callable]:
    """
    Perform window-based soft matching and return merging & unmerging functions.
    """
    B, N, C = input_tensor.shape
    windows_2, _, _ = create_windows(input_tensor, w, h, window_size=2)
    similarity_matrix_2 = calculate_similarity(windows_2)
    merged_windows_2, merge_indices_2 = merge_windows(windows_2, similarity_matrix_2, r)

    windows_8, _, _ = create_windows(input_tensor, w, h, window_size=8)
    similarity_matrix_8 = calculate_similarity(windows_8)
    merged_windows_8, merge_indices_8 = merge_windows(windows_8, similarity_matrix_8, r)

    merged_windows_2 = merged_windows_2.permute(0, 3, 1, 2)
    merged_windows_8 = merged_windows_8.permute(0, 3, 1, 2)
    target_size = (merged_windows_8.shape[2], merged_windows_8.shape[3])  

    merged_windows_2_resized = F.interpolate(merged_windows_2, size=target_size, mode='bilinear', align_corners=False)
    merged_windows_2_final = merged_windows_2_resized.permute(0, 2, 3, 1)
    merged_windows_8_final = merged_windows_8.permute(0, 2, 3, 1)

    result = torch.cat([merged_windows_2_final, merged_windows_8_final], dim=1)  # [B, num_windows * 2, window_size, C]
    result = result.view(B, -1, C)  # ✅ [B, N, C]로 변환

    # weights = torch.tensor([0.4, 0.6], device=input_tensor.device).view(1, 1, 1, 2)
    # merged_x = (weights[..., 0] * merged_windows_2_final + weights[..., 1] * merged_windows_8_final)

    def merge_fn(x):
        return result

    def unmerge_fn(merged_x: torch.Tensor) -> torch.Tensor:
        return unmerge_windows(merged_x, merge_indices_2 + merge_indices_8, h, w)

    return merge_fn, unmerge_fn
