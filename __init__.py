import importlib

# 로드할 노드 파일의 목록을 정의합니다.
# 파일 이름에서 .py 확장자는 제외합니다.
node_list = [
    "unified_generator",
]

# 모든 노드의 클래스와 표시 이름을 담을 딕셔너리를 초기화합니다.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 목록에 있는 각 모듈을 동적으로 불러옵니다.
for module_name in node_list:
    try:
        # from . import unified_generator 와 동일한 효과
        imported_module = importlib.import_module(f".{module_name}", __name__)

        # 불러온 모듈에서 클래스와 이름 매핑 정보를 가져와 합칩니다.
        NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"[Unified-LLM-Node] '{module_name}' 모듈 로딩 중 오류 발생: {e}")


# ComfyUI가 이 변수들을 인식하여 노드를 등록합니다.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# 시작 시 콘솔에 성공 메시지 출력 (선택 사항)
print("\033[92m[Unified-LLM-Node] 🚀 통합 LLM/VLM 생성 노드가 성공적으로 로드되었습니다.\033[0m")