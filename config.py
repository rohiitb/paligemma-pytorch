from enum import Enum

class SiglipVisionConfig(Enum):
    hidden_size = 768
    intermediate_size = 3072
    image_size = 224
    patch_size = 16
    num_hidden_layers = 12,
    num_attention_heads = 12,
    num_channels = 3
    layer_norm_eps = 1e-5
    attention_dropout = 0.0
    num_image_tokens = None