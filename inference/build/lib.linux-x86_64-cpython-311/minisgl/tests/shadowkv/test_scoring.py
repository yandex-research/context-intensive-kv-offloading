import pytest
import torch
from minisgl.kernel.shadowkv import shadowkv_score_landmarks_kernel_hd128

torch.manual_seed(42)

@pytest.mark.parametrize(
    "BS,HC,gqa_factor,num_landmarks,batch_indices",
    [
        (1, 4, 2, [4], [0]),
        (3, 4, 2, [16, 4, 5], [0, 1, 2]),
        (4, 8, 8, [144, 28, 255, 125], [2, 0, 1, 3])
    ]
)
def test_shadowkv_score_kernel_gqa_after_score(BS, HC, gqa_factor, num_landmarks, batch_indices):
    assert len(batch_indices) == BS

    alloc_kwargs = {'device': 'cuda', 'dtype': torch.float32}

    num_qo_heads, HD = HC * gqa_factor, 128

    max_num_landmarks = max(num_landmarks)
    cu_num_landmarks = torch.tensor(num_landmarks, device='cuda', dtype=torch.int32)
    cu_batch_indices = torch.tensor(batch_indices, device='cuda', dtype=torch.int32)

    query_states = torch.rand((BS, num_qo_heads, 1, HD), **alloc_kwargs)
    landmarks = torch.rand((BS, HC, max_num_landmarks, HD), **alloc_kwargs)

    mean_query_states = torch.mean(query_states.view(BS, HC, gqa_factor, 1, HD), dim=2)

    kernel_res = torch.empty((BS, HC, max_num_landmarks), **alloc_kwargs)

    shadowkv_score_landmarks_kernel_hd128(
        mean_query_states,
        kernel_res,
        landmarks,
        HC,
        cu_num_landmarks,
        max_num_landmarks,
        cu_batch_indices,
    )

    for i, batch_idx in enumerate(batch_indices):
        repeated_landmarks = torch.repeat_interleave(landmarks[batch_idx][:, :num_landmarks[batch_idx]], dim=0, repeats=gqa_factor)
        assert repeated_landmarks.shape == (num_qo_heads, num_landmarks[batch_idx], HD)

        scores = torch.matmul(query_states[i], repeated_landmarks.transpose(-1, -2)) # * (HD ** -.5)
        scores = torch.mean(scores.view(HC, gqa_factor, num_landmarks[batch_idx]), dim=1)

        print(scores)
        print(kernel_res[i])

        assert torch.allclose(kernel_res[i][:, :num_landmarks[batch_idx]], scores, atol=1e-12)
        # assert torch.allclose(kernel_res[i][:, :num_landmarks[batch_idx]], scores, atol=0.5)

