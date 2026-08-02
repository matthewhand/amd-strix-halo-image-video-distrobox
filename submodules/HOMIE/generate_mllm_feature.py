import os
import math
import json
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

Qwen3VL_PREFIX = '''
1. You are an expert multimodal AI assistant specialized in highly controllable visual content generation.

2. You will receive A set of images depicting specific humans and objects and a text description detailing the target patterns of human-object interaction.

3. Your task: analyze the precise relationships between the visual entities in the images and the interaction logic defined in the text prompt. Specifically, evaluate how the distinct human and object features from the visual inputs should be spatially, physically, and semantically integrated to accurately reconstruct the interaction described in the text.
'''

# ========================= Formal generation system prompt =========================

from torchvision.transforms import ToPILImage
from qwen_vl_utils import process_vision_info

to_pil = ToPILImage()

class Qwen3VL_Embdder(torch.nn.Module):
    def __init__(self, model_path, max_length=1024, dtype=torch.bfloat16, device='cuda'):
        super(Qwen3VL_Embdder, self).__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        # default: Load the model on the available device(s)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=self.dtype,
        ).to(self.device)
        self.model.requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.prefix = Qwen3VL_PREFIX

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = h_new // 16 * 16
        w_new = w_new // 16 * 16

        img_resized = img.resize((w_new, h_new), Image.LANCZOS)
        return img_resized, img.size
    def preprocess_ref_image(self, img):
        # 1. 打开图片
        img, img_size = self.input_process_image(Image.open(img).convert("RGB")) # resize
        img = self.load_image(img) # to tensor
        return img
    
    def forward(self, caption, ref_images):
        # import pdb;pdb.set_trace()
        embs = torch.zeros(1, self.max_length, 2048, dtype=torch.bfloat16, device=self.device)
        masks = torch.zeros(1, self.max_length, dtype=torch.long, device=self.device)

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        # prefix

        messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})
        # image input
        for img in ref_images:
            messages[0]["content"].append({"type": "image", "image": to_pil(img[0])})
        # 添加 prompt
        messages[0]["content"].append({"type": "text", "text": f"{caption[0]}"})   

        # Preparation for inference
        # import pdb;pdb.set_trace()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        inputs.attention_mask = (inputs.input_ids>0).long().to(self.device)

        # import pdb;pdb.set_trace()

        outputs = self.model(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, pixel_values = inputs.pixel_values.to(self.device), image_grid_thw = inputs.image_grid_thw.to(self.device), output_hidden_states=True)
        emb = outputs.hidden_states[-1]

        # 截取掉vision_start_token_id之前的部分
        start_of_vision_token_id = 151652
        vis_input_start_id = (inputs.input_ids == start_of_vision_token_id).nonzero(as_tuple=True)[1][0]
        embs[0,:min(self.max_length, emb.shape[1]-vis_input_start_id)] = emb[0,vis_input_start_id:][:self.max_length]
        # # hidden_states[idx,:min(self.max_length,hidden_state.shape[1]-217)] = hidden_state[0,217:][:self.max_length]
        masks[0,:min(self.max_length,emb.shape[1]-vis_input_start_id)] = torch.ones((min(self.max_length,emb.shape[1]-vis_input_start_id)), dtype=torch.long, device=self.device)

        return embs, masks

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL MLLM features for HOMIE test samples that lack them."
    )
    parser.add_argument("--meta_file", type=str, required=True,
                        help="Input meta jsonl (reference_paths/prompt per line).")
    parser.add_argument("--output_meta", type=str, required=True,
                        help="Output meta jsonl, same as input with a qwen_feature path added per line.")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory to write the per-sample qwen_feature .pt files.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Qwen3-VL-2B-Thinking checkpoint.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max MLLM feature sequence length.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    qwen3vl_embedder = Qwen3VL_Embdder(
        args.model_path, max_length=args.max_length, device=args.device
    )

    os.makedirs(args.feature_dir, exist_ok=True)

    done = 0
    output_data = []
    with open(args.meta_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            caption = [data.get("prompt")]

            # take all subjects (flatten the per-subject reference_paths lists)
            ref_images = [x for sub in data.get("reference_paths") for x in sub]

            print(ref_images)
            ref_images = [qwen3vl_embedder.preprocess_ref_image(ref_path) for ref_path in ref_images]
            embs, masks = qwen3vl_embedder(caption, ref_images)

            target_file_path = os.path.join(args.feature_dir, f"{line_idx}_multis.pt")
            torch.save(embs.cpu(), target_file_path)

            data.update({"mllm_feature": target_file_path})
            output_data.append(data)

            done += 1
            if done % 100 == 0:
                print(f"processed {done} samples")

    with open(args.output_meta, "w", encoding="utf-8") as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {done} features to {args.feature_dir}; new meta -> {args.output_meta}")