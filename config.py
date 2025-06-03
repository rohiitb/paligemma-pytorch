from dataclasses import dataclass
from typing import Optional

@dataclass
class SiglipVisionConfig:
    hidden_size: int = 768
    projection_dim: int = 2048
    intermediate_size: int = 3072
    image_size: int = 224
    patch_size: int = 16
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None

@dataclass
class GemmaConfig:
    vocab_size: int = 257152
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 12
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 8192
    attention_dropout: float = 0.0
    attention_bias: bool = False
    rope_theta: float = 10000.0
    
@dataclass
class PaliGemmaConfig:
    vision_config: SiglipVisionConfig
    text_config: GemmaConfig
    ignore_index: int = -100
    image_token_index: int = 256000
    vocab_size: int = 257152
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: int = -1
    