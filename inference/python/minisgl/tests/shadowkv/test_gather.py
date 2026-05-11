import pytest
import torch
from minisgl.kernel.shadowkv import shadowkv_gather_kv_cache_kernel_hd128

torch.manual_seed(42)


@pytest.mark.parametrize(
    "BS,HC,page_table,gather_indices,num_tokens_to_gather,max_seq_len,batch_indices",
    [
        (
            1,
            4,
            [[0, 1, 2, 3, 4, 5, 6]],
            [[[0, 1, 2], [3, 2, 1], [3, 4, 5], [0, 1, 3]]],
            [3],
            7,
            [0],
        ),
    ],
)
def test_shadowkv_score_kernel_gqa_after_score(
    BS, HC, page_table, gather_indices, num_tokens_to_gather, max_seq_len, batch_indices
):
    assert len(batch_indices) == BS
    HD = 128

    alloc_kwargs = {"device": "cuda", "dtype": torch.float32}

    cu_page_table = torch.tensor(page_table, device="cuda", dtype=torch.int32)
    cu_gather_indices = torch.tensor(gather_indices, device="cuda", dtype=torch.int64)
    cu_num_tokens_to_gather = torch.tensor(num_tokens_to_gather, device="cuda", dtype=torch.int32)
    cu_batch_indices = torch.tensor(batch_indices, device="cuda", dtype=torch.int32)

    k_cache = torch.rand(BS * max_seq_len, 1, HC, HD, **alloc_kwargs)
    v_cache = torch.rand(BS * max_seq_len, 1, HC, HD, **alloc_kwargs)

    k_cache_res = torch.zeros(BS * max_seq_len, 1, HC, HD, **alloc_kwargs)
    v_cache_res = torch.zeros(BS * max_seq_len, 1, HC, HD, **alloc_kwargs)

    shadowkv_gather_kv_cache_kernel_hd128(
        cu_page_table,
        k_cache,
        v_cache,
        k_cache_res,
        v_cache_res,
        HC,
        cu_gather_indices,
        cu_num_tokens_to_gather,
        cu_batch_indices,
        max_seq_len,
    )

    for i, batch_idx in enumerate(batch_indices):
        for j in range(max_seq_len):
            for k in range(HC):
                actual_res_k = k_cache_res[i * max_seq_len + j][0][k]
                actual_res_v = v_cache_res[i * max_seq_len + j][0][k]
                if j >= num_tokens_to_gather[batch_idx]:
                    expected_k = torch.zeros(HD, **alloc_kwargs)
                    expected_v = torch.zeros(HD, **alloc_kwargs)
                else:
                    page_table_index = gather_indices[batch_idx][k][j]
                    expected_k = k_cache[page_table[batch_idx][page_table_index]][0][k]
                    expected_v = v_cache[page_table[batch_idx][page_table_index]][0][k]

                assert expected_k.shape == (HD,)
                assert expected_v.shape == (HD,)

                assert torch.allclose(actual_res_k, expected_k, atol=1e-12)
                assert torch.allclose(actual_res_v, expected_v, atol=1e-12)
