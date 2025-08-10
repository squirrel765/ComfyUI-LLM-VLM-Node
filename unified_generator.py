import folder_paths
import os
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler, Qwen25VLChatHandler
import base64
from torchvision.transforms import ToPILImage
import gc
import torch

# GGUF 모델 파일이 저장된 폴더를 설정합니다.
supported_LLava_extensions = set(['.gguf'])
llava_checkpoints_path = os.path.join(folder_paths.models_dir, "LLavacheckpoints")

if not os.path.isdir(llava_checkpoints_path):
    os.makedirs(llava_checkpoints_path)

folder_paths.folder_names_and_paths["LLavacheckpoints"] = ([llava_checkpoints_path], supported_LLava_extensions)

class UnifiedGeneratorLP:
    """
    텍스트 전용 LLM과 멀티모달 VLM 기능을 하나의 노드로 통합합니다.
    'mode' 입력을 통해 두 가지 작동 방식을 선택할 수 있습니다.
    - Text-Only: 텍스트 프롬프트만 사용하여 텍스트를 생성합니다.
    - Multimodal: 이미지와 텍스트 프롬프트를 함께 사용하여 텍스트를 생성합니다.
    """
    def __init__(self):
        self.llm = None
        self.clip = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "mode": (["Text-Only", "Multimodal"], {"default": "Text-Only"}),
                "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 128000, "step": 64}),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "unload": ("BOOLEAN", {"default": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_msg": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "LLM/Generators"

    def generate(self, ckpt_name, mode, max_ctx, gpu_layers, n_threads,
                 max_tokens, temperature, top_p, top_k, frequency_penalty,
                 presence_penalty, repeat_penalty, seed, unload, prompt, system_msg,
                 image=None, clip_name=None):

        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        chat_handler = None
        
        if mode == "Multimodal":
            if image is None or clip_name is None:
                raise ValueError("Multimodal mode requires an 'image' and a 'clip_name' input.")

            clip_path = folder_paths.get_full_path("LLavacheckpoints", clip_name)
            
            if "qwen2.5-vl" in str(clip_name).lower():
                self.clip = Qwen25VLChatHandler(clip_model_path=clip_path, verbose=False)
            else:
                self.clip = Llava16ChatHandler(clip_model_path=clip_path, verbose=False)
            chat_handler = self.clip

            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

            messages = [
                {"role": "system", "content": system_msg},
                { "role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": base64_string}},
                    {"type": "text", "text": f"{prompt}"}
                ]}
            ]
            
            if system_msg == "You are a helpful assistant.":
                messages[0]["content"] = "You are an assistant who perfectly describes images."

        else: # Text-Only mode
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            # 여기를 수정했습니다. "chatml" 대신 None을 사용합니다.
            chat_handler = None

        self.llm = Llama(model_path=ckpt_path, chat_handler=chat_handler, offload_kqv=True, f16_kv=True,
                         use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024,
                         verbose=True, seed=seed, n_ctx=max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads,
                         logits_all=True, echo=False)

        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty, seed=seed,
        )
        
        if unload:
            if self.llm is not None:
                self.llm.close()
                del self.llm; self.llm = None
            if self.clip is not None:
                if hasattr(self.clip, '_exit_stack') and self.clip._exit_stack:
                    self.clip._exit_stack.close()
                del self.clip; self.clip = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (f"{response['choices'][0]['message']['content']}", )

# ComfyUI에 노드를 등록하기 위한 설정
NODE_CLASS_MAPPINGS = {
    "UnifiedGenerator|LP": UnifiedGeneratorLP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedGenerator|LP": "Unified LLM/VLM Generator",
}