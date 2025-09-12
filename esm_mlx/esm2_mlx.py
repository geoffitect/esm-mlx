# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MLX implementation of ESM2 transformer for protein language modeling."""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.transformer import TransformerEncoderLayer

from .config import ESM2Config


class ESM2Embeddings(nn.Module):
    """Embeddings for ESM2 - token embeddings only (no positional embeddings)."""
    
    def __init__(self, config: ESM2Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embed_scale = config.embed_scale
    
    def __call__(self, input_ids: mx.array) -> mx.array:
        # Get token embeddings only - ESM2 doesn't use positional embeddings
        embeddings = self.word_embeddings(input_ids) * self.embed_scale
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ESM2AttentionLayer(nn.Module):
    """Multi-head attention layer for ESM2."""
    
    def __init__(self, config: ESM2Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_size = config.hidden_size
        
        # Include bias terms to match PyTorch ESM2
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Scale factor for attention scores (PyTorch calls it 'scaling')
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.scaling = self.scale  # PyTorch compatibility
    
    def __call__(
        self, 
        hidden_states: mx.array, 
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        
        # We receive (batch, seq, dim) but need to work in (seq, batch, dim) like PyTorch
        batch_size, seq_len, embed_dim = hidden_states.shape
        tgt_len = seq_len  # PyTorch naming
        bsz = batch_size   # PyTorch naming
        
        # Convert to seq_first: (batch, seq, dim) -> (seq, batch, dim)
        query = hidden_states.transpose(1, 0, 2)
        
        # PyTorch ESM self-attention: q = self.q_proj(query), etc.
        q = self.q_proj(query)
        k = self.k_proj(query)  
        v = self.v_proj(query)
        
        # PyTorch scaling: q *= self.scaling
        q = q * self.scaling
        
        # PyTorch view and transpose:
        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        # Now: (bsz * num_heads, tgt_len, head_dim)
        
        # PyTorch bmm: attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = mx.matmul(q, k.transpose(0, 2, 1))
        # Result: (bsz * num_heads, tgt_len, tgt_len)
        
        # Apply mask if provided (skip for now to debug core computation)
        if attention_mask is not None:
            mask_expanded = mx.repeat(attention_mask, self.num_heads, axis=0)
            mask_expanded = mask_expanded[:, None, :] * mask_expanded[:, :, None]
            mask_value = mx.array(-1e9)
            attn_weights = mx.where(mask_expanded == 1, attn_weights, mask_value)
        
        # PyTorch softmax
        attn_probs = mx.softmax(attn_weights, axis=-1)
        # Skip dropout: attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # PyTorch bmm: attn = torch.bmm(attn_probs, v)
        attn = mx.matmul(attn_probs, v)
        # Result: (bsz * num_heads, tgt_len, head_dim)
        
        # PyTorch final reshape: attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = attn.transpose(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        # Result: (tgt_len, bsz, embed_dim) = (seq, batch, dim)
        
        # PyTorch output projection: attn = self.out_proj(attn)
        attn = self.out_proj(attn)
        
        # Convert back to batch_first for our model: (seq, batch, dim) -> (batch, seq, dim)
        attn_output = attn.transpose(1, 0, 2)
        
        # For output_attentions, average over heads
        output_attn_weights = None
        if output_attentions:
            attn_probs_reshaped = attn_probs.reshape(bsz, self.num_heads, tgt_len, tgt_len)
            output_attn_weights = mx.mean(attn_probs_reshaped, axis=1)
        
        return attn_output, output_attn_weights


class ESM2TransformerLayer(nn.Module):
    """Single transformer layer for ESM2 matching PyTorch architecture."""
    
    def __init__(self, config: ESM2Config):
        super().__init__()
        self.attention = ESM2AttentionLayer(config)
        
        # PyTorch ESM2 uses these specific names and post-layer norm
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Add aliases for weight conversion compatibility
        self.layer_norm1 = self.self_attn_layer_norm
        self.layer_norm2 = self.final_layer_norm
        
        # Feed forward network with bias terms
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        
        # Self-attention with PRE-layer norm (PyTorch ESM2 style)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer norm BEFORE attention
        attn_output, attn_weights = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        hidden_states = residual + attn_output  # Residual connection
        
        # FFN with PRE-layer norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Layer norm BEFORE FFN
        hidden_states = self.fc1(hidden_states)
        # GELU activation - use exact PyTorch implementation
        # GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
        # Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        hidden_states = hidden_states * 0.5 * (1.0 + mx.erf(hidden_states / mx.sqrt(2.0)))
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states  # Residual connection
        
        return hidden_states, attn_weights


class ESM2Encoder(nn.Module):
    """Stack of ESM2 transformer layers."""
    
    def __init__(self, config: ESM2Config):
        super().__init__()
        self.layers = [ESM2TransformerLayer(config) for _ in range(config.num_hidden_layers)]
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        
        all_hidden_states = [hidden_states] if output_hidden_states else []
        all_attentions = [] if output_attentions else []
        
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, attention_mask, output_attentions)
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_attentions if output_attentions else None
        }


class ESM2MLX(nn.Module):
    """MLX implementation of ESM2 for protein language modeling."""
    
    def __init__(self, config: ESM2Config):
        super().__init__()
        self.config = config
        
        # Core components
        self.embeddings = ESM2Embeddings(config)
        self.encoder = ESM2Encoder(config)
        
        # Optional components
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_contacts: bool = False
    ) -> Dict[str, Any]:
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask,
            output_attentions,
            output_hidden_states
        )
        
        sequence_output = encoder_outputs["last_hidden_state"]
        
        # Language modeling head
        prediction_scores = self.lm_head(sequence_output)
        
        # Contact prediction (simplified version)
        contacts = None
        if return_contacts and output_attentions:
            # Average attention weights across layers and heads for contact prediction
            attentions = encoder_outputs["attentions"]
            if attentions:
                # Stack and average
                stacked_attentions = mx.stack(attentions, axis=0)  # [num_layers, batch, heads, seq, seq]
                contacts = mx.mean(stacked_attentions, axis=(0, 2))  # Average over layers and heads
        
        return {
            "prediction_scores": prediction_scores,
            "last_hidden_state": sequence_output,
            "hidden_states": encoder_outputs.get("hidden_states"),
            "attentions": encoder_outputs.get("attentions"), 
            "contacts": contacts
        }
    
    @classmethod
    def from_config(cls, config: ESM2Config):
        """Create model from configuration."""
        return cls(config)
    
    def get_sequence_embeddings(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Get sequence-level embeddings by mean pooling over sequence length."""
        outputs = self(input_ids, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]
        
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.astype(hidden_states.dtype)[..., None]
            hidden_states = hidden_states * mask_expanded
            # Average over valid tokens
            seq_lengths = mx.sum(attention_mask, axis=1, keepdims=True)
            embeddings = mx.sum(hidden_states, axis=1) / seq_lengths
        else:
            # Simple mean pooling
            embeddings = mx.mean(hidden_states, axis=1)
        
        return embeddings