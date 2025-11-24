# ComfyUI 통합 LLM/VLM 노드

이 저장소는 텍스트 기반 LLM(거대 언어 모델)과 멀티모달 VLM(비전 언어 모델)을 하나의 노드로 통합한 ComfyUI 커스텀 노드를 포함합니다. 이 노드를 사용하면 ComfyUI 워크플로우 내에서 직접 텍스트 생성 및 이미지 설명(이미지-to-텍스트) 작업을 수행할 수 있습니다.
이 프로젝트는 LevelPixel/ComfyUI-LevelPixel-Advanced의 코드 구조에서 영감을 받아 제작되었습니다. 원본 개발자에게 감사를 표합니다.


## ✨ 주요 기능
1. **통합 노드**: 텍스트 생성과 이미지 기반 생성을 위한 단일 노드를 제공합니다.
2. **듀얼 모드**: 'Text-Only'와 'Multimodal' 두 가지 모드 간 전환이 가능합니다.
3. **GGUF 모델 지원**: llama-cpp-python을 기반으로 하여 인기 있는 GGUF 모델 형식을 사용합니다.
4. **VLM 지원**: Llava 1.6 및 Qwen2.5-VL, Qwen3.0-VL과 같은 비전 모델과 호환되어 이미지를 이해하고 설명할 수 있습니다.
5. **메모리 관리**: 생성 후 VRAM 및 RAM을 확보할 수 있는 'Unload' 옵션이 포함되어 있습니다.
6. **세부 설정**: Temperature, Top_p, Top_k, 페널티 등 다양한 생성 파라미터를 정밀하게 제어할 수 있습니다.

## 📦 설치 방법
**반드시 ComfyUI에 내장된 Python 환경에 llama-cpp-python을 설치해야 합니다.**<br>
[llama-cpp-python.git](https://github.com/JamePeng/llama-cpp-python.git)

```
# example
pip install -U --force-reinstall https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.16-cu128-AVX2-win-20251112/llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl
```
```
cd ComfyUI/custom_nodes/
git clone https://github.com/squirrel765/ComfyUI-LLM-VLM-Node.git
```

## 🚀 사용 방법
### 모델 파일 준비
**LLM/VLM 모델**: .gguf 형식의 주 모델 파일을 **ComfyUI/models/LLavacheckpoints/** 폴더에 넣으세요. 폴더가 없다면 새로 생성해야 합니다. <br>

### 추천모델<br>

#### 멀티모달 모델
[thesby/Qwen2.5-VL-7B-NSFW-Caption-V3](https://huggingface.co/thesby/Qwen2.5-VL-7B-NSFW-Caption-V3)
- **gguf**모델과 함께 **mmproj**-model-f16.gguf 같이 이름에 **mmproj** 가 들어간 clip모델도 다운 받아주세요
- (25.11.16) **QWEN 3.0 8B gguf** 모델의 정상작동을 확인하였습니다. 


#### 텍스트 모델

[Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M) <br>

[nidum/Nidum-Llama-3.2-3B-Uncensored-GGUF](https://huggingface.co/nidum/Nidum-Llama-3.2-3B-Uncensored-GGUF) <br>

**CLIP 모델 (멀티모달용)**: 멀티모달 모드에서 사용할 비전 모델 파일(예: mmproj-model-f16.gguf)도 같은 **ComfyUI/models/LLavacheckpoints/** 폴더에 넣으세요. 

### 노드 사용법
ComfyUI 메뉴에서 Add Node > LLM/Generators > Unified LLM/VLM Generator를 추가합니다.

### 모드 (Text-Only , Multimodal):

**Text-Only**: 텍스트 프롬프트만 사용하여 텍스트를 생성합니다. <br>

**Multimodal**: 이미지와 텍스트 프롬프트를 함께 사용하여 텍스트를 생성합니다. <br>

---

**ckpt_name**: LLavacheckpoints 폴더에 있는 주 GGUF 모델을 선택합니다. <br>
**image (멀티모달 전용)**: 다른 노드(예: Load Image)의 이미지 출력을 연결합니다. <br>
**clip_name (멀티모달 전용)**: ckpt_name에 맞는 비전(CLIP) 모델을 선택합니다. <br>
**prompt**: 모델에 전달할 텍스트 프롬프트입니다. <br>
**system_msg**: 모델의 역할이나 행동을 정의하는 시스템 프롬프트입니다. <br>
**gpu_layers**: VRAM에 오프로드할 레이어 수를 지정합니다. 사용자의 VRAM 크기에 맞게 조절하세요. (-1은 모든 레이어를 오프로드합니다.) <br>
**unload**: True로 설정하면 생성 작업이 끝난 후 모델을 메모리에서 해제하여 시스템 리소스를 절약합니다. <br>

**max_ctx (Maximum Context Size)** : LLM이 한 번에 기억하고 처리할 수 있는 최대 토큰 수(입력 + 출력 포함). <br>
**gpu_layers** : 모델의 레이어 중 몇 개를 GPU에서 실행할지 지정. <br>
**n_threads** : LLM 연산을 수행할 때 사용할 CPU 스레드 수. <br>
**max_tokens** : 모델이 생성할 수 있는 최대 출력 토큰 수. <br>
**temperature** : 출력의 무작위성(창의성) 을 조절하는 값 (0~2) <br>
**top_p (Nucleus Sampling)** : 상위 확률의 토큰 누적 확률이 top_p가 될 때까지의 후보만 고려해 샘플링. <br>
**top_k** : 매 스텝마다 확률이 높은 상위 k개의 토큰 중 하나를 선택. <br>

## 예시
<img width="1324" height="730" alt="Image" src="https://github.com/user-attachments/assets/7444d711-ba08-483a-aaff-db5831f8f8f3" /> <br>
<img width="1354" height="767" alt="Image" src="https://github.com/user-attachments/assets/d8e34624-fa94-41c6-a61c-1e08a7b705bd" /> <br>
<img width="1589" height="774" alt="Image" src="https://github.com/user-attachments/assets/a02300e4-941f-493d-890d-82d252e427a2" /> <br>
