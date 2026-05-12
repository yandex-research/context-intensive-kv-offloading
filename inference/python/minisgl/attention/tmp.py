def compute_and_store_landmarks(
    self,
    key_states: torch.Tensor,
    layer_idx: int,
    cu_seqlens_k: torch.Tensor,
    batch_indices: list[int],
):
    # assert len(batch_indices) == 1, key_states.shape

    # if layer_idx == 0:
    #     print(cu_seqlens_k)
    #     print(key_states.shape)

    for i, batch_index in enumerate(batch_indices):
        cu_seqlen = int(cu_seqlens_k[i])

        # batch_index = batch_indices[0]
        num_chunks = self.total_num_chunks[batch_index]

        if num_chunks == 0:
            continue

        key_states_loc = key_states[
            cu_seqlen
            + self.prefix_end_indices[batch_index] : cu_seqlen
            + self.suffix_start_indices[batch_index]
        ]
        SL = key_states_loc.shape[0]
        assert (
            SL > 0
        ), f"{key_states.shape} {cu_seqlen=} {cu_seqlens_k=} {i=} {batch_indices=} {self.prefix_end_indices[batch_index]} {self.suffix_start_indices[batch_index]}"
        # print(layer_idx, SL, SL % self.config.chunk_size)

        key_states_loc = key_states_loc.view(
            SL, self.local_kv_heads, self.model_config.head_dim
        ).transpose(0, 1)

        if self.config.chunk_size > 1:
            key_states_loc = key_states_loc.view(
                self.local_kv_heads, num_chunks, self.config.chunk_size, self.model_config.head_dim
            )

            # if layer_idx == 3:
            #     print(key_states_loc.shape)
            #     print(self.landmarks_buffer[layer_idx, batch_index])

            new_landmarks = torch.mean(key_states_loc, dim=2)
        else:
            new_landmarks = key_states_loc

        self.landmarks_buffer[layer_idx, batch_index].index_copy_(
            dim=1, index=torch.arange(num_chunks, device=self.device), source=new_landmarks
        )

    # if layer_idx == 3:
    #     print(self.landmarks_buffer[layer_idx, batch_index])
