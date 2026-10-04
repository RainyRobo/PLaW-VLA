import types

import torch

from openpi.models_pytorch.gemma_pytorch import _multi_stream_attention_forward
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def _make_attention_module(attn_implementation: str, *, num_key_value_groups: int):
    return types.SimpleNamespace(
        num_key_value_groups=num_key_value_groups,
        training=False,
        config=types.SimpleNamespace(_attn_implementation=attn_implementation),
    )


def test_multi_stream_sdpa_matches_eager_on_block_mask_forward_and_backward():
    torch.manual_seed(0)

    batch_size = 2
    num_query_heads = 8
    num_kv_heads = 4
    num_key_value_groups = num_query_heads // num_kv_heads
    query_len = 6
    key_len = 6
    head_dim = 16

    pad_masks = torch.ones((batch_size, key_len), dtype=torch.bool)
    att_masks = torch.tensor(
        [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    attention_mask = torch.where(
        att_2d_masks[:, None, :, :],
        torch.zeros((), dtype=torch.float32),
        torch.full((), -1.0e9, dtype=torch.float32),
    )

    eager_module = _make_attention_module("eager", num_key_value_groups=num_key_value_groups)
    sdpa_module = _make_attention_module("sdpa", num_key_value_groups=num_key_value_groups)

    query = torch.randn((batch_size, num_query_heads, query_len, head_dim), dtype=torch.float32)
    key = torch.randn((batch_size, num_kv_heads, key_len, head_dim), dtype=torch.float32)
    value = torch.randn((batch_size, num_kv_heads, key_len, head_dim), dtype=torch.float32)

    eager_query = query.clone().requires_grad_(True)
    eager_key = key.clone().requires_grad_(True)
    eager_value = value.clone().requires_grad_(True)
    sdpa_query = query.clone().requires_grad_(True)
    sdpa_key = key.clone().requires_grad_(True)
    sdpa_value = value.clone().requires_grad_(True)

    eager_out, _ = _multi_stream_attention_forward(
        eager_module,
        eager_query,
        eager_key,
        eager_value,
        attention_mask,
        scaling=head_dim**-0.5,
    )
    sdpa_out, _ = _multi_stream_attention_forward(
        sdpa_module,
        sdpa_query,
        sdpa_key,
        sdpa_value,
        attention_mask,
        scaling=head_dim**-0.5,
    )

    torch.testing.assert_close(sdpa_out, eager_out, atol=1e-5, rtol=1e-5)

    eager_out.sum().backward()
    sdpa_out.sum().backward()

    torch.testing.assert_close(sdpa_query.grad, eager_query.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sdpa_key.grad, eager_key.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sdpa_value.grad, eager_value.grad, atol=1e-5, rtol=1e-5)
