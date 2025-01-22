import torch
import torch.nn.functional as F
from typing import Tuple, Callable
import math

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
    B, num_windows, window_size, C = windows.shape
    similarity_matrix = similarity_matrix.clone()
    similarity_matrix[:, torch.arange(num_windows), torch.arange(num_windows)] = float('-inf')  # 자기 자신 제외
    r_windows = int(num_windows * 0.4)  # 입력 텐서 기반이 아니라, 윈도우 개수 기준으로 변환
    valid_r = min(r_windows, num_windows // 2)  # 동일한 기준으로 비교
    # valid_r = min(r, num_windows // 2, similarity_matrix.shape[-1]) 
    print("valid_r: ",valid_r)
    print("r: ",r_windows)
    # ✅ 선택된 윈도우를 담을 리스트
    selected_idx1 = []
    selected_idx2 = []
    num_selected = 0
    print(f"num_selected: {num_selected}, valid_r: {valid_r}")
    while num_selected < valid_r:
        values, indices = similarity_matrix.view(B, -1).topk(valid_r - num_selected, dim=-1)  # 아직 필요한 개수만큼만 가져오기
        new_idx1 = indices // num_windows
        new_idx2 = indices % num_windows
        print("new idx1: ",new_idx1.shape)
        print("new idx2: ",new_idx2.shape)
        # ✅ 중복 제거
        # mask = ~torch.isin(new_idx1, torch.tensor(selected_idx1, device=similarity_matrix.device)) & \
        #     ~torch.isin(new_idx2, torch.tensor(selected_idx2, device=similarity_matrix.device))
        mask1 = ~torch.isin(new_idx1, torch.tensor(selected_idx1, device=similarity_matrix.device).unsqueeze(0))
        mask2 = ~torch.isin(new_idx2, torch.tensor(selected_idx2, device=similarity_matrix.device).unsqueeze(0))

        # ✅ batch-wise AND 연산 적용
        mask = mask1 & mask2  # 이제 (2, 128) 차원 유지됨
        new_idx1 = new_idx1[mask].view(B, -1)
        new_idx2 = new_idx2[mask].view(B, -1)
        print("after 중복제거 --------------------")
        print("new idx1: ",new_idx1.shape)
        print("new idx2: ",new_idx2.shape)
        # ✅ 리스트에 추가하면서 정확히 128개를 유지
        selected_idx1.extend(new_idx1.tolist())
        selected_idx2.extend(new_idx2.tolist())
        num_selected = len(selected_idx1[1])  # 업데이트
        print("num_selected: ",num_selected)

    # ✅ 최종 선택된 인덱스를 텐서로 변환
    idx1 = torch.tensor(selected_idx1, device=similarity_matrix.device).view(B, -1)
    idx2 = torch.tensor(selected_idx2, device=similarity_matrix.device).view(B, -1)

    print(f"idx1 unique count: {len(torch.unique(idx1))}")
    print(f"idx2 unique count: {len(torch.unique(idx2))}")
    print(f"idx1.shape before filtering: {idx1.shape}")
    print(f"idx2.shape before filtering: {idx2.shape}") 
    # merged = (windows[:, idx1] + windows[:, idx2]) / 2  # 두 윈도우 평균 내기
    unique_pairs = idx1 != idx2  # 같은 윈도우가 선택되지 않도록 필터링
    idx1_filtered = [torch.masked_select(idx1[b], unique_pairs[b]) for b in range(B)]
    idx2_filtered = [torch.masked_select(idx2[b], unique_pairs[b]) for b in range(B)]
    idx1_final = torch.cat(idx1_filtered).view(B, -1)
    idx2_final = torch.cat(idx2_filtered).view(B, -1)

    print(f"idx1.shape after filtering: {idx1_final.shape}")
    print(f"idx2.shape after filtering: {idx2_final.shape}")
    unique_idx2, inverse_indices, counts = torch.unique(idx2_final, return_inverse=True, return_counts=True)
    duplicate_idx2 = unique_idx2[counts > 1]  # 중복된 idx2 찾기
    if unique_idx2.shape[0] > valid_r:
        unique_idx2 = unique_idx2[:valid_r]  # valid_r 개만 유지
    print(f"Unique idx2 count after fixing: {unique_idx2.shape[0]} (Expected: {valid_r})")

    # ✅ 병합 수행
    merged = (windows[:, idx1_final] + windows[:, idx2_final]) / 2  
    merged = merged.mean(dim=1)  # ✅ 차원 조정
    actual_r = idx1_final.shape[1]  # 실제 병합된 개수 반영
    print(f"actual_r after filtering: {actual_r}")
    if actual_r == 0:
        print("⚠ Warning: No valid merge pairs found. Skipping merge step.")
        return windows

    # ✅ 기존 윈도우에서 제거할 윈도우 제외
    mask = torch.ones(num_windows, device=windows.device, dtype=torch.bool)
    idx2_final_unique = torch.unique(idx2_final)  # 중복 제거
    print("idx2_final_unique : ", idx2_final_unique)
    mask[idx2_final_unique.flatten()] = False  # 제거할 윈도우 False로 설정
    print(f"mask.sum(): {mask.sum()}")
    print("unique_idx2 : ", unique_idx2)
    # 🔥 추가 수정: unique_idx2와 duplicate_idx2도 False로 설정
    mask[unique_idx2] = False  
    print(f"mask.sum(): {mask.sum()}")
    print("duplicate_idx2 : ", duplicate_idx2)
    mask[duplicate_idx2] = False  

    '''
    mask[idx2_final.flatten()] = False  
    # unique_idx2, counts = torch.unique(idx2_final, return_counts=True)
    mask[unique_idx2] = False  # 중복 제거된 idx2만 False 처리
    mask[duplicate_idx2] = False  # 🛑 추가: 중복된 idx2도 False로 처리  
    '''
    
    print(f"mask.sum(): {mask.sum()} (Should be {num_windows - actual_r})")
    remaining_windows = windows[:, mask]
    print(f"remaining_windows.shape: {remaining_windows.shape}")

    print(f"merged.shape before fixing: {merged.shape}")  

    merged_windows = merged
    print(f"Windows after merging shape: {merged_windows.shape}, mean: {merged_windows.mean()}")

    return merged_windows, actual_r

def reconstruct_from_windows(windows: torch.Tensor, num_windows_h: int, num_windows_w: int, 
                             window_size: int, num_windows: int) -> torch.Tensor:
    """
    윈도우 형태를 원래 이미지 공간으로 복원하는 함수.

    Args:
        windows (torch.Tensor): 윈도우 텐서 [B, num_windows, W*W, C]
        num_windows_h (int): 높이 방향 윈도우 개수
        num_windows_w (int): 너비 방향 윈도우 개수
        window_size (int): 개별 윈도우 크기
        num_windows (int): 실제 윈도우 개수

    Returns:
        torch.Tensor: 복원된 이미지 [B, N, C] 형태
    """
    B, actual_num_windows, _, C = windows.shape

    # ✅ `num_windows_h * num_windows_w`가 `num_windows`와 다를 경우 조정
    if num_windows_h * num_windows_w != num_windows:
        print(f"⚠ Warning: Adjusting num_windows_h and num_windows_w. Expected: {num_windows}, Actual: {num_windows_h * num_windows_w}")
        
        # 가장 가까운 num_windows_h와 num_windows_w 찾기
        factor_pairs = [(h, num_windows // h) for h in range(1, num_windows + 1) if num_windows % h == 0]
        best_pair = min(factor_pairs, key=lambda p: abs(p[0] - p[1]))  # 가장 비율이 비슷한 조합 선택
        num_windows_h, num_windows_w = best_pair

        print(f"🔄 Adjusted num_windows_h: {num_windows_h}, num_windows_w: {num_windows_w}")

    # ✅ `view()`를 사용하기 전에 크기 맞추기
    try:
        windows = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
    except RuntimeError as e:
        print(f"❌ view() 실패: {e}, windows.shape: {windows.shape}, num_windows_h: {num_windows_h}, num_windows_w: {num_windows_w}")
        return windows  # 오류 발생 시 원본 윈도우 반환

    # ✅ 윈도우를 다시 이미지 형태로 복원
    reconstructed = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    reconstructed = reconstructed.view(B, num_windows_h * window_size, num_windows_w * window_size, C)

    # ✅ `[B, N, C]` 형태로 변환
    return reconstructed.view(B, num_windows_h * window_size * num_windows_w * window_size, C)

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

    # **🔹 Step 1: 4×4 윈도우 생성 및 Merge 수행**
    windows_4, num_windows_h_4, num_windows_w_4 = create_windows(input_tensor, w, h, window_size=4)
    similarity_matrix_4 = calculate_similarity(windows_4)
    print(f"Before merging: {windows_4.shape}")  # 병합 전
    merged_windows_4, numw4 = merge_windows(windows_4, similarity_matrix_4, r, mode)
    print(f"After merging: {merged_windows_4.shape}")  # 병합 후
    print(f"merged_windows_4.shape: {merged_windows_4.shape}")
    # reconstructed_4 = reconstruct_from_windows(merged_windows_4, num_windows_h_4, num_windows_w_4, 4, h, w)
    reconstructed_4 = reconstruct_from_windows(merged_windows_4, num_windows_h_4, num_windows_w_4, 4, numw4)

    # **🔹 Step 2: 16×16 윈도우 생성 및 Merge 수행**
    windows_16, num_windows_h_16, num_windows_w_16 = create_windows(input_tensor, w, h, window_size=16)
    similarity_matrix_16 = calculate_similarity(windows_16)
    merged_windows_16, numw16 = merge_windows(windows_16, similarity_matrix_16, r, mode)
    # reconstructed_16 = reconstruct_from_windows(merged_windows_16, num_windows_h_16, num_windows_w_16, 16, h, w)
    reconstructed_16 = reconstruct_from_windows(merged_windows_16, num_windows_h_16, num_windows_w_16, 16, numw16)

    # **🔹 Step 3: Interpolation을 통해 크기 맞추기 (이제야 수행)**
    reconstructed_4_resized = F.interpolate(reconstructed_4.permute(0, 2, 1), size=N, mode='linear').permute(0, 2, 1)
    reconstructed_16_resized = F.interpolate(reconstructed_16.permute(0, 2, 1), size=N, mode='linear').permute(0, 2, 1)

    # **🔹 Step 4: Concatenation 수행**
    result = torch.cat([reconstructed_4_resized, reconstructed_16_resized], dim=-1)

    print(f"result_direct.shape: {result.shape}")  # 이 과정이 정상 동작하는지 확인

    return result
