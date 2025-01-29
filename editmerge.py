
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

# def calculate_similarity(windows: torch.Tensor) -> torch.Tensor:
#     """
#     Calculate cosine similarity between windows.

#     Args:
#         windows (torch.Tensor): Windows tensor [B, num_windows, W*W, C].

#     Returns:
#         torch.Tensor: Similarity matrix [B, num_windows, num_windows].
#     """
#     window_features = windows.mean(dim=2)  # [B, num_windows, C]
#     # 정규화된 특징을 한 번에 계산
#     norm = torch.norm(window_features, dim=-1, keepdim=True)
#     normalized_features = window_features / (norm + 1e-6)
#     # 배치 행렬 곱셈 사용
#     return torch.bmm(normalized_features, normalized_features.transpose(-2, -1))
# def merge_windows(windows: torch.Tensor, similarity_matrix: torch.Tensor, r: int, mode="mean") -> torch.Tensor:
#     B, num_windows, window_size, C = windows.shape

#     # ✅ r 값을 similarity_matrix 크기 내로 제한
#     valid_r = min(r, similarity_matrix.shape[-1])

#     # ✅ 상위 r개의 유사한 쌍 선택
#     values, indices = similarity_matrix.view(B, -1).topk(valid_r, dim=-1)
#     idx1 = indices // num_windows
#     idx2 = indices % num_windows

#     print(f"Windows before merging mean: {windows.mean()}")

#     # ✅ scatter_() 대신 직접 값 변경 (gather 사용)
#     merged_windows = windows.clone()
#     if mode == "mean":
#         merged = (windows.gather(1, idx1.unsqueeze(-1).expand(-1, -1, window_size, C)) + 
#                   windows.gather(1, idx2.unsqueeze(-1).expand(-1, -1, window_size, C))) / 2
#         merged_windows = merged  # 직접 할당하여 변경 반영

#     print(f"Windows after merging mean: {merged_windows.mean()}")

#     return merged_windows
#     # B, num_windows, window_size, C = windows.shape
#     # valid_r = min(r, similarity_matrix.shape[-1])

#     # # 상위 r개의 유사한 쌍 선택
#     # values, indices = similarity_matrix.view(B, -1).topk(valid_r, dim=-1)
#     # idx1 = indices // num_windows
#     # idx2 = indices % num_windows

#     # # 벡터화된 병합 연산
#     # merged_windows = windows.clone()
#     # if mode == "mean":
#     #     # 인덱싱을 사용한 벡터화된 연산
#     #     merged = (windows[torch.arange(B).unsqueeze(1), idx1] + 
#     #               windows[torch.arange(B).unsqueeze(1), idx2]) / 2
#     #     merged_windows.scatter_(1, idx1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size, C), merged)

#     # return merged_windows
#     # # 상위 r개의 유사한 쌍 선택
#     # values, indices = similarity_matrix.view(B, -1).topk(r, dim=-1)
#     # idx1 = indices // num_windows
#     # idx2 = indices % num_windows
    
#     # # 벡터화된 병합 연산
#     # merged_windows = windows.clone()
#     # if mode == "mean":
#     #     # 인덱싱을 사용한 벡터화된 연산
#     #     merged = (windows[torch.arange(B).unsqueeze(1), idx1] + 
#     #              windows[torch.arange(B).unsqueeze(1), idx2]) / 2
#     #     merged_windows.scatter_(1, idx1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size, C), merged)
    
#     # return merged_windows

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

# def merge_windows(windows: torch.Tensor, similarity_matrix: torch.Tensor, r: int, mode="mean") -> torch.Tensor:
#     """
#     Merge similar windows using similarity matrix.

#     Args:
#         windows (torch.Tensor): Windows tensor [B, num_windows, window_size, C].
#         similarity_matrix (torch.Tensor): Similarity matrix [B, num_windows, num_windows].
#         r (int): Number of merges to perform.
#         mode (str): Merge mode ('mean' or 'sum').

#     Returns:
#         torch.Tensor: Merged windows tensor.
#     """
#     B, num_windows, window_size, C = windows.shape

#     # ✅ Merge 먼저 수행
#     valid_r = min(r, similarity_matrix.shape[-1])
#     values, indices = similarity_matrix.view(B, -1).topk(valid_r, dim=-1)
#     idx1 = indices // num_windows
#     idx2 = indices % num_windows

#     merged_windows = windows.clone()

#     if mode == "mean":
#         # idx1_expanded = idx1.unsqueeze(-1).repeat(1, 1, window_size).unsqueeze(-1)
#         # idx2_expanded = idx2.unsqueeze(-1).repeat(1, 1, window_size).unsqueeze(-1)
#         idx1_expanded = idx1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size, C)
#         idx2_expanded = idx2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size, C)
#         merged = (windows.gather(1, idx1_expanded) + windows.gather(1, idx2_expanded)) / 2
#         merged_windows = merged  # 직접 할당하여 변경 반영

#     print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

#     return merged_windows
import torch

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
        # ✅ 중복 제거
        # mask = ~torch.isin(new_idx1, torch.tensor(selected_idx1, device=similarity_matrix.device)) & \
        #     ~torch.isin(new_idx2, torch.tensor(selected_idx2, device=similarity_matrix.device))
        # mask1 = ~torch.isin(new_idx1, torch.tensor(selected_idx1, device=similarity_matrix.device).unsqueeze(0))
        # mask2 = ~torch.isin(new_idx2, torch.tensor(selected_idx2, device=similarity_matrix.device).unsqueeze(0))

    #     # ✅ batch-wise AND 연산 적용
    #     mask = mask1 & mask2  # 이제 (2, 128) 차원 유지됨
    #     new_idx1 = new_idx1[mask].view(B, -1)
    #     new_idx2 = new_idx2[mask].view(B, -1)
    #     print("after 중복제거 --------------------")
    #     print("new idx1: ",new_idx1.shape)
    #     print("new idx2: ",new_idx2.shape)
    #     # ✅ 리스트에 추가하면서 정확히 128개를 유지
    #     selected_idx1.extend(new_idx1.tolist())
    #     selected_idx2.extend(new_idx2.tolist())
    #     num_selected = len(selected_idx1[1])  # 업데이트
    #     print("num_selected: ",num_selected)

    # # ✅ 최종 선택된 인덱스를 텐서로 변환
    # idx1 = torch.tensor(selected_idx1, device=similarity_matrix.device).view(B, -1)
    # idx2 = torch.tensor(selected_idx2, device=similarity_matrix.device).view(B, -1)

    # print(f"idx1 unique count: {len(torch.unique(idx1))}")
    # print(f"idx2 unique count: {len(torch.unique(idx2))}")
    # print(f"idx1.shape before filtering: {idx1.shape}")
    # print(f"idx2.shape before filtering: {idx2.shape}") 
    # # merged = (windows[:, idx1] + windows[:, idx2]) / 2  # 두 윈도우 평균 내기
    # unique_pairs = idx1 != idx2  # 같은 윈도우가 선택되지 않도록 필터링
    # idx1_filtered = [torch.masked_select(idx1[b], unique_pairs[b]) for b in range(B)]
    # idx2_filtered = [torch.masked_select(idx2[b], unique_pairs[b]) for b in range(B)]
    # idx1_final = torch.cat(idx1_filtered).view(B, -1)
    # idx2_final = torch.cat(idx2_filtered).view(B, -1)

    # print(f"idx1.shape after filtering: {idx1_final.shape}")
    # print(f"idx2.shape after filtering: {idx2_final.shape}")
    # unique_idx2, inverse_indices, counts = torch.unique(idx2_final, return_inverse=True, return_counts=True)
    # duplicate_idx2 = unique_idx2[counts > 1]  # 중복된 idx2 찾기
    # if unique_idx2.shape[0] > valid_r:
    #     unique_idx2 = unique_idx2[:valid_r]  # valid_r 개만 유지
    # print(f"Unique idx2 count after fixing: {unique_idx2.shape[0]} (Expected: {valid_r})")

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
    # merged = (windows[:, idx1_final] + windows[:, idx2_final]) / 2
    # merged = merged.mean(dim=1)
    # # merged = (windows.gather(1, idx1_final.unsqueeze(-1).expand(-1, -1, window_size, C)) + 
    # #           windows.gather(1, idx2_final.unsqueeze(-1).expand(-1, -1, window_size, C))) / 2

    # actual_r = idx1_final.shape[1]  # 실제 병합된 개수 반영
    # print(f"actual_r after filtering: {actual_r}")

    # if actual_r == 0:
    #     print("⚠ Warning: No valid merge pairs found. Skipping merge step.")
    #     return windows

    # ✅ 기존 윈도우에서 제거할 윈도우 제외
    # mask = torch.ones(num_windows, device=windows.device, dtype=torch.bool)
    # idx2_final_unique = torch.unique(idx2_final)  # 중복 제거
    # print("idx2_final_unique : ", idx2_final_unique)
    # mask[idx2_final_unique.flatten()] = False  # 제거할 윈도우 False로 설정
    # print(f"mask.sum(): {mask.sum()}")
    # print("unique_idx2 : ", unique_idx2)
    # # 🔥 추가 수정: unique_idx2와 duplicate_idx2도 False로 설정
    # mask[unique_idx2] = False  
    # print(f"mask.sum(): {mask.sum()}")
    # print("duplicate_idx2 : ", duplicate_idx2)
    # mask[duplicate_idx2] = False  

    '''
    mask[idx2_final.flatten()] = False  
    # unique_idx2, counts = torch.unique(idx2_final, return_counts=True)
    mask[unique_idx2] = False  # 중복 제거된 idx2만 False 처리
    mask[duplicate_idx2] = False  # 🛑 추가: 중복된 idx2도 False로 처리  
    '''
    
    # print(f"mask.sum(): {mask.sum()} (Should be {num_windows - actual_r})")
    # remaining_windows = windows[:, mask]
    # print(f"remaining_windows.shape: {remaining_windows.shape}")

    # print(f"merged.shape before fixing: {merged.shape}")  
    # ✅ 병합된 윈도우와 남은 윈도우 합치기
    #merged_windows = torch.cat([remaining_windows, merged], dim=1)

    merged_windows = merged
    # print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

    return merged_windows, actual_r
# # 🛠️ 디버깅 출력
#     print(f"Unique idx2 count: {unique_idx2.shape[0]} (Expected: {idx2.numel()})")
#     print(f"Duplicate indices count: {(counts > 1).sum().item()}")  # 몇 개의 인덱스가 중복되었는지 출력
#     # mask[idx2_final.flatten()] = False  # idx2에 해당하는 윈도우를 제거
#     mask[unique_idx2] = False 
#     print(f"mask.sum(): {mask.sum()} (Should be less than {num_windows})")
#     remaining_windows = windows[:, mask]
#     print(f"remaining_windows.shape: {remaining_windows.shape}")

#     print(f"merged.shape before fixing: {merged.shape}")  
#     # ✅ 병합된 윈도우와 남은 윈도우 합치기
#     merged_windows = torch.cat([remaining_windows, merged], dim=1)

#     print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

#     return merged_windows
    # # 다시 리스트를 텐서로 변환 (길이가 다를 수 있으므로 패딩 필요)
    # max_len = max(x.shape[0] for x in idx1_filtered) if idx1_filtered else 0
    # idx1_padded = torch.stack([torch.cat([x, torch.full((max_len - x.shape[0],), -1, dtype=torch.long, device=x.device)]) for x in idx1_filtered])
    # idx2_padded = torch.stack([torch.cat([x, torch.full((max_len - x.shape[0],), -1, dtype=torch.long, device=x.device)]) for x in idx2_filtered])

    # print(f"idx1.shape after filtering: {idx1_padded.shape}")  # Expected: [B, new_valid_r]
    # print(f"idx2.shape after filtering: {idx2_padded.shape}")
    # merged = (windows[:, idx1_padded] + windows[:, idx2_padded]) / 2  # 올바른 병합 수행
    # # ✅ unique_pairs로 필터링한 후 실제 r값을 다시 계산
    # actual_r = idx1_padded.shape[1] if idx1_padded.dim() == 2 else idx1_padded.shape[0]
    # print(f"actual_r after filtering: {actual_r}")


    # # ✅ unique_pairs로 필터링한 후 실제 r값을 다시 계산
    # # actual_r = idx1.shape[1]  # 필터링된 병합 개수
    # if idx1.dim() == 1:  # idx1이 1차원일 경우
    #     actual_r = idx1.shape[0]  # 첫 번째 차원을 사용
    # else:
    #     actual_r = idx1.shape[1]  # 기존 방식 유지
    # print(f"actual_r after filtering: {actual_r}")

    # if actual_r == 0:
    #     print("⚠ Warning: No valid merge pairs found. Skipping merge step.")
    #     return windows  # 병합 없이 원본 반환

    # merged = merged[:, :actual_r]  # 실제 병합 개수에 맞게 자르기

    # mask = torch.ones(num_windows, device=windows.device, dtype=torch.bool)
    # mask[idx2.flatten()] = False  # idx2에 해당하는 윈도우를 제거

    
    # remaining_windows = windows[:, mask]  # 제거된 윈도우 제외
    # print("valid_r :  ", actual_r)
    # print("merged.shape before view:", merged.shape)
    # merged = merged.view(B, actual_r, window_size, C)  # valid_r 대신 actual_r 사용
    # # merged = merged[:, unique_pairs] ////////////////
    # # merged = merged.mean(dim=1)  # ✅ 두 번째 차원 제거
    # # # ✅ 선택된 윈도우 병합
    # # if mode == "mean":
    # #     merged = (windows[:, idx1] + windows[:, idx2]) / 2  # 두 윈도우 평균 내기
    # # elif mode == "sum":
    # #     merged = windows[:, idx1] + windows[:, idx2]  # 두 윈도우 더하기
    # # else:
    # #     raise ValueError("Invalid mode. Choose 'mean' or 'sum'.")

    # # ✅ 기존 윈도우에서 제거할 윈도우 제외
    # # mask = torch.ones(num_windows, device=windows.device, dtype=torch.bool)
    # # mask[idx2.flatten()] = False  # idx2에 해당하는 윈도우를 제거

    # # ✅ 남은 윈도우들만 선택
    # # remaining_windows = windows[:, mask]  # 제거된 윈도우 제외
    # # print("valid_r :  ", valid_r)
    # # print("merged.shape before view:", merged.shape)
    # # merged = merged.view(B, valid_r, window_size, C)  # valid_r 차원 제거
    
    # # ✅ 병합된 윈도우와 남은 윈도우 합치기
    # merged_windows = torch.cat([remaining_windows, merged], dim=1)  # 병합된 윈도우 추가

    # print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

    # return merged_windows  # 최종 윈도우 개수: num_windows - r

# def reconstruct_from_windows(windows: torch.Tensor, num_windows_h: int, num_windows_w: int, window_size: int, h: int, w: int) -> torch.Tensor:
#     """
#     Reconstruct the image from windows.

#     Args:
#         windows (torch.Tensor): Windows tensor [B, num_windows, W*W, C].
#         num_windows_h (int): Number of windows along height.
#         num_windows_w (int): Number of windows along width.
#         window_size (int): Size of the window.
#         h (int): Original height of the image.
#         w (int): Original width of the image.

#     Returns:
#         torch.Tensor: Reconstructed image tensor [B, N, C].
#     """
#     B, num_windows, _, C = windows.shape

#     # Reshape windows: [B, num_windows_h, num_windows_w, window_size, window_size, C]
#     windows = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)

#     # Combine windows back into the image
#     reconstructed = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
#     reconstructed = reconstructed.view(B, h, w, C)
#     return reconstructed.view(B, h * w, C)
import torch
import math

# def reconstruct_from_windows(windows: torch.Tensor, num_windows_h: int, num_windows_w: int, window_size: int, h: int, w: int) -> torch.Tensor:
#     """
#     Merge된 윈도우를 다시 원본 크기 (h, w)로 복원.

#     Args:
#         windows (torch.Tensor): Windows tensor [B, num_windows, W*W, C].
#         num_windows_h (int): 병합 후 남은 윈도우의 높이 개수.
#         num_windows_w (int): 병합 후 남은 윈도우의 너비 개수.
#         window_size (int): 각 윈도우 크기.
#         h (int): 원본 이미지 높이.
#         w (int): 원본 이미지 너비.

#     Returns:
#         torch.Tensor: Reconstructed image tensor [B, h*w, C].
#     """
#     B, num_windows, _, C = windows.shape

#     # ⚠ 동적으로 num_windows_h, num_windows_w 조정
#     num_windows_h = math.isqrt(num_windows)  # 정사각형에 가까운 값
#     num_windows_w = num_windows // num_windows_h  # 너비 계산

#     if num_windows_h * num_windows_w != num_windows:
#         print(f"⚠ Warning: Adjusting num_windows_h and num_windows_w. Expected: {num_windows}, Actual: {num_windows_h * num_windows_w}")
#         return windows  # 원본 반환

#     # ✅ 윈도우를 원래 공간으로 복원
#     windows = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
#     reconstructed = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
#     reconstructed = reconstructed.view(B, num_windows_h * window_size, num_windows_w * window_size, C)
#     # return reconstructed
#     return reconstructed.view(B, num_windows_h * window_size * num_windows_w * window_size, C)
#----
# def reconstruct_from_windows(windows: torch.Tensor, num_windows_h: int, num_windows_w: int, 
#                              window_size: int, num_windows: int) -> torch.Tensor:
#     """
#     윈도우 형태를 원래 이미지 공간으로 복원하는 함수.

#     Args:
#         windows (torch.Tensor): 윈도우 텐서 [B, num_windows, W*W, C]
#         num_windows_h (int): 높이 방향 윈도우 개수
#         num_windows_w (int): 너비 방향 윈도우 개수
#         window_size (int): 개별 윈도우 크기
#         num_windows (int): 실제 윈도우 개수

#     Returns:
#         torch.Tensor: 복원된 이미지 [B, N, C] 형태
#     """
#     B, actual_num_windows, _, C = windows.shape

#     # ✅ `num_windows_h * num_windows_w`가 `num_windows`와 다를 경우 조정
#     if num_windows_h * num_windows_w != num_windows:
#         print(f"⚠ Warning: Adjusting num_windows_h and num_windows_w. Expected: {num_windows}, Actual: {num_windows_h * num_windows_w}")
        
#         # 가장 가까운 num_windows_h와 num_windows_w 찾기
#         factor_pairs = [(h, num_windows // h) for h in range(1, num_windows + 1) if num_windows % h == 0]
#         best_pair = min(factor_pairs, key=lambda p: abs(p[0] - p[1]))  # 가장 비율이 비슷한 조합 선택
#         num_windows_h, num_windows_w = best_pair

#         print(f"🔄 Adjusted num_windows_h: {num_windows_h}, num_windows_w: {num_windows_w}")

#     # ✅ `view()`를 사용하기 전에 크기 맞추기
#     try:
#         windows = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
#     except RuntimeError as e:
#         print(f"❌ view() 실패: {e}, windows.shape: {windows.shape}, num_windows_h: {num_windows_h}, num_windows_w: {num_windows_w}")
#         return windows  # 오류 발생 시 원본 윈도우 반환

#     # ✅ 윈도우를 다시 이미지 형태로 복원
#     reconstructed = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
#     reconstructed = reconstructed.view(B, num_windows_h * window_size, num_windows_w * window_size, C)

#     # ✅ `[B, N, C]` 형태로 변환
#     return reconstructed.view(B, num_windows_h * window_size * num_windows_w * window_size, C)
#---
import torch
import torch.nn.functional as F

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
    # if merged_windows_4.shape[1] != merged_windows_16.shape[1]:
    #     merged_windows_16 = merged_windows_16.permute(0, 3, 1, 2)
    #     target_size = (merged_windows_4.shape[1], merged_windows_4.shape[2])  # (102, 16)
    #     merged_windows_16_resized = F.interpolate(merged_windows_16, 
    #                                       size=target_size,  # (num_windows, height)
    #                                       mode='bilinear', align_corners=False)  # Bilinear Interpolation 사용

    # merged_windows_16_final = merged_windows_16_resized.permute(0, 2, 3, 1)
    # result = torch.cat([merged_windows_4, merged_windows_16_final], dim=-1)
    

    # def unmerge_fn(merged_x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Unmerge function to revert back to the original tensor shape.
    #     """
    #     # Concatenation 이전의 개별 윈도우 크기를 알아야 원래 상태로 복구 가능
    #     recovered_windows_4 = merged_x[..., :numw4, :]
    #     recovered_windows_16 = merged_x[..., numw4:, :]

    #     # Interpolation을 통해 원래 크기 복원
    #     recovered_windows_16 = F.interpolate(
    #         recovered_windows_16.permute(0, 3, 1, 2), 
    #         size=(h, w), mode='bilinear', align_corners=False
    #     ).permute(0, 2, 3, 1)

    #     # 원래 윈도우 크기대로 복구
    #     reconstructed = (recovered_windows_4 + recovered_windows_16) / 2  # 단순 평균 복구 (조정 가능)
    #     return reconstructed

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
#         window_size (int): Size of the window.
#         r (int): Number of merges to perform.
#         mode (str): Merge mode ('mean' or 'sum').

#     Returns:
#         torch.Tensor: Reconstructed tensor [B, N, C].
#     """
#     B, N, C = input_tensor.shape
#     device = input_tensor.device

#     # Process with 4x4 windows
#     windows_4, num_windows_h_4, num_windows_w_4 = create_windows(input_tensor, w, h, window_size=4)
#     similarity_matrix_4 = calculate_similarity(windows_4)
#     merged_windows_4 = merge_windows(windows_4, similarity_matrix_4, r, mode)
#     reconstructed_4 = reconstruct_from_windows(merged_windows_4, num_windows_h_4, num_windows_w_4, 4, h, w)

#     # Process with 16x16 windows
#     windows_16, num_windows_h_16, num_windows_w_16 = create_windows(input_tensor, w, h, window_size=16)
#     similarity_matrix_16 = calculate_similarity(windows_16)
#     merged_windows_16 = merge_windows(windows_16, similarity_matrix_16, r, mode)
#     reconstructed_16 = reconstruct_from_windows(merged_windows_16, num_windows_h_16, num_windows_w_16, 16, h, w)

#     print("Windows 4 shape:", windows_4.shape)
#     print("Windows 16 shape:", windows_16.shape)

#     print("Similarity matrix 4 mean:", similarity_matrix_4.mean())
#     print("Similarity matrix 16 mean:", similarity_matrix_16.mean())

#     print("Windows 4 before merging mean:", windows_4.mean())
#     print("Merged Windows 4 mean:", merged_windows_4.mean())

#     print("Windows 16 before merging mean:", windows_16.mean())
#     print("Merged Windows 16 mean:", merged_windows_16.mean())

#     print("Merging 4 changed?", not torch.equal(windows_4, merged_windows_4))
#     print("Merging 16 changed?", not torch.equal(windows_16, merged_windows_16))



#     # Concatenate results along the channel dimension
#     result = torch.cat([reconstructed_4, reconstructed_16], dim=-1)
    
#     return result
#------원본
# import torch
# from typing import Tuple, Callable


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


# def bipartite_soft_matching_random2d(metric: torch.Tensor,
#                                      w: int, h: int, sx: int, sy: int, r: int,
#                                      no_rand: bool = False,
#                                      generator: torch.Generator = None) -> Tuple[Callable, Callable]:
#     """
#     Partitions the tokens into src and dst and merges r tokens from src to dst.
#     Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

#     Args:
#      - metric [B, N, C]: metric to use for similarity
#      - w: image width in tokens
#      - h: image height in tokens
#      - sx: stride in the x dimension for dst, must divide w
#      - sy: stride in the y dimension for dst, must divide h
#      - r: number of tokens to remove (by merging)
#      - no_rand: if true, disable randomness (use top left corner only)
#      - rand_seed: if no_rand is false, and if not None, sets random seed.
#     """
#     B, N, _ = metric.shape

#     if r <= 0:
#         return do_nothing, do_nothing

#     gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
#     with torch.no_grad():
#         hsy, wsx = h // sy, w // sx

#         # For each sy by sx kernel, randomly assign one token to be dst and the rest src
#         if no_rand:
#             rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
#         else:
#             rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
#         # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
#         idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
#         idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
#         idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

#         # Image is not divisible by sx or sy so we need to move it into a new buffer
#         if (hsy * sy) < h or (wsx * sx) < w:
#             idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
#             idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
#         else:
#             idx_buffer = idx_buffer_view

#         # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
#         rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

#         # We're finished with these
#         del idx_buffer, idx_buffer_view

#         # rand_idx is currently dst|src, so split them
#         num_dst = hsy * wsx
#         a_idx = rand_idx[:, num_dst:, :] # src
#         b_idx = rand_idx[:, :num_dst, :] # dst

#         def split(x):
#             C = x.shape[-1]
#             src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
#             dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
#             return src, dst

#         # Cosine similarity between A and B
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = split(metric)
#         scores = a @ b.transpose(-1, -2)

#         # Can't reduce more than the # tokens in src
#         r = min(a.shape[1], r)

#         # Find the most similar greedily
#         node_max, node_idx = scores.max(dim=-1)
#         edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

#         unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
#         src_idx = edge_idx[..., :r, :]  # Merged Tokens
#         dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = split(x)
#         n, t1, c = src.shape
        
#         unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
#         src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

#         return torch.cat([unm, dst], dim=1)

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         unm_len = unm_idx.shape[1]
#         unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
#         _, _, c = unm.shape

#         src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

#         # Combine back to the original shape
#         out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
#         out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
#         out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
#         out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

#         return out

#     return merge, unmerge
