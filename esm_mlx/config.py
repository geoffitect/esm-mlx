# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

@dataclass
class ESM2Config:
    """Configuration for ESM2 MLX model."""
    vocab_size: int = 33
    hidden_size: int = 1280
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1026
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
    mask_token_id: int = 32
    cls_token_id: int = 0
    eos_token_id: int = 2
    
    # ESM2 specific
    embed_scale: float = 1.0
    attention_head_dim: int = 64  # hidden_size // num_attention_heads
    use_rotary_embeddings: bool = False
    
    def __post_init__(self):
        if self.attention_head_dim is None:
            self.attention_head_dim = self.hidden_size // self.num_attention_heads