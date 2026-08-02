# Text encoder wrapper: UMT5EncoderModel + AutoTokenizer.
#
# Reproduces the NPU WanPipeline._get_prompt_embeds (wan_pipeline.py:581-614):
#   * tokenize with padding="max_length", max_length=text_len (512), truncation,
#     add_special_tokens, return attention mask,
#   * run UMT5 encoder -> last_hidden_state (bf16),
#   * trim each sample to its real length (mask sum) then zero-pad back to text_len.
import os

import torch


class HomieT5:
    def __init__(self, ckpt_dir, text_encoder_subfolder="text_encoder",
                 tokenizer_subfolder="tokenizer", text_len=512,
                 dtype=torch.bfloat16, device="cuda", shard_fn=None):
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.device = device
        self.dtype = dtype
        self.text_len = text_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ckpt_dir, tokenizer_subfolder)
        )
        self.model = UMT5EncoderModel.from_pretrained(
            os.path.join(ckpt_dir, text_encoder_subfolder), torch_dtype=dtype
        )
        self.model.eval().requires_grad_(False)
        if shard_fn is not None:
            self.model = shard_fn(self.model)
        else:
            self.model.to(device)

    @torch.no_grad()
    def __call__(self, prompts, device=None):
        """prompts: List[str]. Returns [B, text_len, hidden] bf16."""
        device = device or self.device
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        embeds = self.model(input_ids, mask).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device=device)
        # trim to real length, then zero-pad back to text_len (matches NPU)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in embeds],
            dim=0,
        )
        return embeds
