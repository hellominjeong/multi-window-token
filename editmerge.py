
import torch
from typing import Tuple, Callable

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


import torch
import torch.nn.functional as F

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
    B, num_windows, window_size, C = windows.shape
    
    similarity_matrix = similarity_matrix.clone()
    similarity_matrix[:, torch.arange(num_windows), torch.arange(num_windows)] = float('-inf')  # 자기 자신 제외
    r_windows = int(num_windows * 0.5)  # 입력 텐서 기반이 아니라, 윈도우 개수 기준으로 변환
    valid_r = min(r_windows, num_windows // 2)  # 동일한 기준으로 비교

    # ✅ 선택된 윈도우를 담을 리스트
    selected_idx1 = []
    selected_idx2 = []
    num_selected = 0

    while num_selected < valid_r:
        # print("in while ---- sim matrix before view", similarity_matrix.shape)
        values, indices = similarity_matrix.view(B, -1).topk(valid_r - num_selected, dim=-1)  # 아직 필요한 개수만큼만 가져오기
        new_idx1 = indices // num_windows
        new_idx2 = indices % num_windows
        # print("new idx1: ",new_idx1.shape)
        # print("new idx2: ",new_idx2.shape)

        valid_pairs = new_idx1 != new_idx2 
        # print("valid_pairs",valid_pairs.shape)
        valid_pairs = valid_pairs.view(B, -1)
        # print("valid_pairs.view",valid_pairs.shape)
        # new_idx1 = new_idx1[valid_pairs]
        
        # new_idx2 = new_idx2[valid_pairs]
        new_idx1 = torch.where(valid_pairs, new_idx1, -1)  # -1을 채워서 구조 유지
        new_idx2 = torch.where(valid_pairs, new_idx2, -1)

        valid_range = (new_idx1 < num_windows) & (new_idx2 < num_windows)
        # print("valid_range",valid_range.shape)
        valid_range = valid_range.view(B, -1)  # 배치 차원 유
        # print("valid_range.view",valid_range.shape)
        # new_idx1 = new_idx1[valid_range]
        # new_idx2 = new_idx2[valid_range]
        new_idx1 = torch.where(valid_range, new_idx1, -1)
          # -1을 채워서 구조 유지
        new_idx2 = torch.where(valid_range, new_idx2, -1)
        # print("Filtered new_idx1: ", new_idx1.shape)
        # print("Filtered new_idx2: ", new_idx2.shape)

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
    # print(f"merged.shape: {merged.shape}")
    # print(f"merged mean: {merged.mean()}, merged shape after mean: {merged.shape}")
    actual_r = new_idx1.shape[1]  # 실제 병합된 개수 반영
    # print(f"actual_r after filtering: {actual_r}")
    if actual_r == 0:
        print("⚠ Warning: No valid merge pairs found. Skipping merge step.")
        return windows

    merged_windows = merged

    return merged_windows, actual_r

import torch
import math

def window_based_soft_matching(
    input_tensor: torch.Tensor,
    w: int,
    h: int,
    r: int,
    mode="mean"
) -> torch.Tensor:
    """
    Perform window-based soft matching and merging.

    Args:
        input_tensor (torch.Tensor): Input tensor [B, N, C].
        w (int): Width of the image in tokens.
        h (int): Height of the image in tokens.
        r (int): Number of merges to perform.
        mode (str): Merge mode ('mean' or 'sum').

    Returns:
        torch.Tensor: Reconstructed tensor [B, N, C].
    """
    B, N, C = input_tensor.shape
    device = input_tensor.device
    # print("input tensor: ===========================")
    # print(input_tensor.shape)
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
    
    def unmerge_fn(merged_x: torch.Tensor, numw4: int, numw8: int, numw16: int, h: int, w: int) -> torch.Tensor:
        """
        Unmerge function to revert back to the original tensor shape.
        """
        # 1. 병합된 텐서를 분리
        # 4x4 윈도우 복구
        recovered_windows_4 = merged_x[..., :, :320]
        
        # 8x8 윈도우는 원래 형태로 유지
        recovered_windows_8 = merged_x[..., :, 320:640]
        
        # 16x16 윈도우 복구
        recovered_windows_16 = merged_x[..., :,  :640]

        print("recovered_windows_4: ",recovered_windows_4.shape)
        print("recovered_windows_8: ",recovered_windows_8.shape)
        print("recovered_windows_16: ",recovered_windows_16.shape)

        # 2. 4x4와 16x16 윈도우 크기를 원래의 크기인 8x8로 보간
        recovered_windows_4_resized = F.interpolate(
            recovered_windows_4.permute(0, 3, 1, 2),  # [B, C, num_windows_h, num_windows_w]
            size=(16, numw4),  # 8x8 윈도우 크기
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # [B, num_windows_h, num_windows_w, C]
        
        recovered_windows_16_resized = F.interpolate(
            recovered_windows_16.permute(0, 3, 1, 2),  # [B, C, num_windows_h, num_windows_w]
            size=(h//16, numw16),  # 8x8 윈도우 크기
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # [B, num_windows_h, num_windows_w, C]
        print("recovered_windows_4_resized: ",recovered_windows_4_resized.shape)
        print("recovered_windows_16_resized: ",recovered_windows_16_resized.shape)
        # 3. 모든 윈도우를 결합
        reconstructed = (
            recovered_windows_4_resized + recovered_windows_8 + recovered_windows_16_resized
        ) / 3  # 단순 평균으로 복원
        print(type(reconstructed))
        
        return reconstructed
    a = unmerge_fn(result, numw4, numw8, numw16, h, w)
    print("---------------", type(a))
    return result, a 

