import json
from transformers import AutoTokenizer
from pathlib import Path
from safetensors import safe_open
from config import PaliGemmaConfig
from gemma import PaliGemmaConditionalGeneration

# Write a fuinction which loads the parameters from the json file into the Config class
def load_config_from_json(config_class, json_file_path):
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)
    return config_class(**config_dict)

def load_huggingface_model(model_path: str, device: str):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right", "Tokenizer padding side must be right"

    # Get the safetensors
    safetensors_path = Path(model_path) / "pytorch_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors file not found at {safetensors_path}")
    
    tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Load the config
    config = load_config_from_json(PaliGemmaConfig, Path(model_path) / "config.json")

    # Initialize the model
    model = PaliGemmaConditionalGeneration(config).to(device)

    # Load the state dict
    model.load_state_dict(tensors, strict=False)

    # Tie weights of embedding and lm_head
    model.tie_weights()

    return model, tokenizer