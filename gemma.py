import torch
from torch import nn
import math
from typing import Optional
import torch.nn.functional as F

from config import PaliGemmaConfig, SiglipVisionConfig, GemmaConfig
from siglip import SiglipVisionModel

class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    @property
    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        # [Batch_size, Num_key_value_heads, Seq_len, Head_dim]
        return self.key_cache[0].shape[-2]
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            # Initialize the cache for the first time
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, position_ids, seq_len=None):
        # Inv freq to x device
        self.inv_freq.to(x.device)
        # Expand inv_freq to batch dim
        # [Dim / 2] -> [1, Dim / 2, 1] -> [Batch_size, Dim / 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        # Expand position_ids
        # [Batch_size, Seq_len] -> [Batch_size, 1, Seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=True):
            # Multiply position_ids_expanded by inv_freq_expanded
            # [Batch_size, Dim / 2, 1] * [Batch_size, 1, Seq_len] -> [Batch_size, Seq_len, Dim / 2]
            freqs = torch.matmul(inv_freq_expanded.float(), position_ids_expanded.float()).transpose(-1, -2)
            # [Batch_size, Seq_len, Dim / 2] -> [Batch_size, Seq_len, Dim]
            embs = torch.cat((freqs, freqs), dim=-1)
            # [Batch_size, Seq_len, Dim]
            cos = embs.cos()
            sin = embs.sin()

        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    # NOTE: This implementation is a bit different from the original implementation
    # Huggingface uses a different implementation of the rotary embedding
    # But the results are the same
    # Build [-x2, x1, -x4, x3, ...]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):
    # Add head dim
    q = q.unsqueeze(unsqueeze_dim)
    k = k.unsqueeze(unsqueeze_dim)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        # Equivalent to
        # y = F.gelu(self.gate_proj(x), approximate="tanh")) # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Intermediate_size]
        # j = self.up_proj(x) # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Intermediate_size]  
        # z = y * j # [Batch_size, Seq_len, Intermediate_size]
        # z = self.down_proj(z) # [Batch_size, Seq_len, Intermediate_size] -> [Batch_size, Seq_len, Hidden_size]
        # [Batch_size, Seq_len, Hidden_size]
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    B, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(B, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(B, num_key_value_heads * n_rep, seq_len, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        bsz, q_len, _ = hidden_states.shape
        # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Num_attention_heads * Head_dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Num_key_value_heads * Head_dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Num_key_value_heads * Head_dim]
        value_states = self.v_proj(hidden_states)
        # [Batch_size, Seq_len, Num_attention_heads * Head_dim] -> [Batch_size, Num_attention_heads, Seq_len, Head_dim]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        # [Batch_size, Seq_len, Num_key_value_heads * Head_dim] -> [Batch_size, Num_key_value_heads, Seq_len, Head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        # [Batch_size, Seq_len, Num_key_value_heads * Head_dim] -> [Batch_size, Num_key_value_heads, Seq_len, Head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

        # [Batch_size, Seq_len, Head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_size, Num_attention_heads, Seq_len, Head_dim]
        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # [Batch_size, Num_key_value_heads, Seq_len, Head_dim] -> [Batch_size, Num_attention_heads, Seq_len, Head_dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # [Batch_size, Num_key_value_heads, Seq_len, Head_dim] -> [Batch_size, Num_attention_heads, Seq_len, Head_dim]
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        assert attention_mask is not None, "Attention mask is required"
        attn_weights = attn_weights + attention_mask


        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # [Batch_size, Num_attention_heads, Seq_len, Seq_len] -> [Batch_size, Num_attention_heads, Seq_len, Head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_attention_heads, q_len, self.head_dim):
            raise ValueError(f"Attention output size {attn_output.size()} does not match expected size {bsz, self.num_attention_heads, q_len, self.head_dim}")

        # [Batch_size, Num_attention_heads, Seq_len, Head_dim] -> [Batch_size, Seq_len, Num_attention_heads, Head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_size, Seq_len, Num_attention_heads, Head_dim] -> [Batch_size, Seq_len, Num_attention_heads * Head_dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # [Batch_size, Seq_len, Num_attention_heads * Head_dim] -> [Batch_size, Seq_len, Hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.to(x.dtype)
    

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.self_attn = GemmaAttention(config)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, kv_cache: Optional[KVCache] = None):
        residual = hidden_states
        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states, attention_mask, position_ids, kv_cache
        )

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = residual + hidden_states

        # [Batch_size, Seq_len, Hidden_size]
        residual = hidden_states
        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        # [Batch_size, Seq_len, Embed_dim]
        hidden_states = inputs_embeds

        normalizer = math.sqrt(self.config.hidden_size)
        hidden_states = hidden_states / normalizer

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, position_ids, kv_cache
            )

        # [Batch_size, Seq_len, Embed_dim]
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states).float()

        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            # Update the kv cache
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Projection_dim]
        hidden_states = self.linear(pixel_values)
        return hidden_states


class PaliGemmaConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = config.pad_token_id

    def tie_weighta(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Normalize the image features
        scaled_image_features = image_features / math.sqrt(
            self.config.vision_config.hidden_size
        )

        # Create final embeddings for combining image tokens and text tokens
        final_embedding = torch.zeros(
            batch_size, seq_len, embed_dim, device=device, dtype=dtype
        )

        # Shape: [Batch_size, Seq_len]
        text_mask = (
            input_ids != self.pad_token_id
            and input_ids != self.config.image_token_index
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Shape: [Batch_size, Seq_len, Embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings to the final embedding
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        # Insert the image embeddings to the final embedding
        # NOTE: Cannot use torch.where because the shape of the tensors are different
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        # Zero the padding tokens
        final_embedding = final_embedding.masked_fill(pad_mask_expanded, 0)

        # Attention mask
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase, no need to mask anything
            # Only works if we have no padding
            causal_mask = torch.zeros(
                (batch_size, q_len, q_len), device=device, dtype=dtype
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Again we dont need to mask anything as the query needs to access all previous tokens
            causal_mask = torch.zeros(
                (batch_size, q_len, kv_len), device=device, dtype=dtype
            )

        # Add head dimension
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None or kv_cache.num_items() > 0:
            # position of the query is the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create position ids based on the attention mask
            # For masked tokens, use number 1 as position
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill(attention_mask == 0, 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        assert torch.all(attention_mask == 1), "Input cannot be padded"

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # [B, C, H, W] -> [B, Num_patches, Embed_dim]
        selected_image_features = self.vision_tower(
            pixel_values.to(inputs_embeds.dtype)
        )
        # [B, Num_patches, Embed_dim] -> [B, Num_patches, Hidden_size]
        image_features = self.multi_modal_projector(selected_image_features)

        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
