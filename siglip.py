import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.position_embeddings = nn.Embedding(
            self.num_patches, config.hidden_size
        )

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor):
        batch_size, num_channels, height, width = pixel_values.shape
        patch_embeds = self.patch_embeddings(pixel_values)
        embeddings = patch_embeds.flatten(2).permute(0, 2, 1)
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipVisionAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.size()
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        query_states = self.q_proj(hidden_states)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        key_states = self.k_proj(hidden_states)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        value_states = self.v_proj(hidden_states)

        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query_states = query_states.permute(0, 2, 1, 3)

        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.permute(0, 2, 1, 3)

        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        value_states = value_states.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipVisionEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.attention = SiglipVisionAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.encoder = SiglipVisionEncoder(config)
        self.embeddings = SiglipVisionEmbeddings(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor):
        embeddings = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embeddings)
        return self.post_layernorm(encoder_outputs)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values [Batch_size, Chanels, Height, Width] -> [Batch_size, Num_patches, Embed_dim]
        return self.vision_model(pixel_values)
