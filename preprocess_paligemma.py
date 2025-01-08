import torch
from typing import List
from PIL import Image
import numpy as np

from preprocess_image_utils import preprocess_image

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def build_string_from_input(prefix_prompt, bos_token, image_seq_len, image_token, num_images):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [
        f"<seg{i:0>3}>" for i in range(128)
    ]

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        tokenizer.add_tokens(self.EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ):
        assert (
            len(images) == 1 and len(text) == 1
        ), "Only one image and one text are supported for inference"

        pixel_values = preprocess_image(
            images,
            image_size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_MEAN,
            image_std=IMAGENET_STD,
        )

        pixel_values = np.stack(pixel_values, axis=0)

        pixel_values = torch.from_numpy(pixel_values)

        input_strings = [
            build_string_from_input(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.num_image_tokens,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
