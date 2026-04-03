"""명함 레이아웃 시각 테스트 스크립트

ComfyUI/VLLM 없이 흰색 배경 PIL Image로 _build_layout() → _render_card_sync_with_plan()
파이프라인을 실행하고, 10개 레이아웃 결과를 output/ 폴더에 PNG로 저장한다.

실행 예시:
    python scripts/test_layouts.py                          # 전체 10개
    python scripts/test_layouts.py --layout split           # 특정 레이아웃
    python scripts/test_layouts.py --layout split two_column
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import types
from pathlib import Path

# ──────────────────────────────────────────────
# 프로젝트 루트를 sys.path에 추가
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────
# app.clients.vllm_client 목(mock) 등록
# card_layout_planner.py 가 import 시 바로 참조하므로
# 실제 모듈 임포트 전에 미리 sys.modules 에 등록해야 한다.
# ──────────────────────────────────────────────
def _register_vllm_mock() -> None:
    # app, app.clients 는 실제 패키지 디렉터리가 있으므로 sys.modules 에 직접
    # 등록하지 않는다. vllm_client 모듈만 가짜로 채워 ImportError 를 방지한다.
    mock_mod = types.ModuleType("app.clients.vllm_client")

    class _DummyClient:
        def __init__(self, *a, **kw):
            self.client = None

    mock_mod.VLLMClient = _DummyClient
    mock_mod.VLMClient  = _DummyClient

    sys.modules["app.clients.vllm_client"] = mock_mod


_register_vllm_mock()

# ──────────────────────────────────────────────
# 실제 모듈 임포트
# ──────────────────────────────────────────────
from PIL import Image  # noqa: E402

from app.services.card_layout_planner import (  # noqa: E402
    CardLayoutPlan,
    _build_layout,
)
from app.services.card_renderer import _render_card_sync_with_plan  # noqa: E402

# ──────────────────────────────────────────────
# 샘플 데이터 (자유롭게 수정 가능)
# ──────────────────────────────────────────────
SAMPLE_CARD = {
    "name":       "홍길동",
    "position":   "백엔드 개발자",
    "company":    "카카오",
    "department": "AI개발팀",
    "phone":      "010-1234-5678",
    "email":      "hong@kakao.com",
    "address":    "서울시 강남구 테헤란로 123",
}

# ──────────────────────────────────────────────
# 레이아웃 테스트 파라미터
# ──────────────────────────────────────────────
CARD_WIDTH  = 1104
CARD_HEIGHT = 624

# 텍스트 영역: 카드 전체에서 여백(80px)을 뺀 넉넉한 사각형
TEXT_REGION = dict(x1=80, y1=80, x2=1024, y2=544)

# 밝은 흰색 배경 기준 다크 텍스트 팔레트
NAME_COLOR    = (8,   8,   8)
INFO_COLOR    = (40,  40,  40)
SUB_COLOR     = (70,  70,  70)
DIVIDER_COLOR = (145, 145, 145)

TEXT_SIZE = "medium"

ALL_LAYOUTS = [
    "stack",
    "split",
    "two_column",
    "name_hero",
    "centered",
    "brand_left",
    "name_right",
    "top_banner",
    "minimal",
    "bottom_name",
]

# ──────────────────────────────────────────────
# 헬퍼: 흰색 PIL Image → base64 data URL
# ──────────────────────────────────────────────
def _white_image_data_url(width: int = CARD_WIDTH, height: int = CARD_HEIGHT) -> str:
    """흰색 배경 PIL Image를 PNG base64 data URL로 변환한다."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────
# 단일 레이아웃 렌더링
# ──────────────────────────────────────────────
def render_layout(layout_style: str, output_dir: Path) -> Path:
    """지정한 layout_style 로 명함을 렌더링해 PNG 파일로 저장하고 경로를 반환한다."""
    tr = TEXT_REGION

    # 1. _build_layout() 으로 FieldLayout dict + DividerLayout 생성
    fields, divider = _build_layout(
        layout_style,
        tr["x1"], tr["y1"], tr["x2"], tr["y2"],
        NAME_COLOR, INFO_COLOR, SUB_COLOR, DIVIDER_COLOR,
        text_size=TEXT_SIZE,
    )

    # 2. CardLayoutPlan 구성 (sdxl_prompt 불필요, shapes 없음)
    plan = CardLayoutPlan(
        sdxl_prompt="",
        sdxl_negative_prompt="",
        fields=fields,
        shapes=[],
        divider=divider,
        style_tag="Test",
        layout_style=layout_style,
    )

    # 3. 흰색 배경 data URL 생성
    bg_data_url = _white_image_data_url()

    # 4. 동기 렌더링 → base64 data URL 반환
    result_data_url = _render_card_sync_with_plan(bg_data_url, SAMPLE_CARD, plan)

    # 5. base64 디코딩 → PIL Image → PNG 저장
    if "," in result_data_url:
        _, encoded = result_data_url.split(",", 1)
    else:
        encoded = result_data_url
    img_bytes = base64.b64decode(encoded)
    result_img = Image.open(io.BytesIO(img_bytes))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"layout_{layout_style}.png"
    result_img.save(out_path, format="PNG")
    return out_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="명함 레이아웃 시각 테스트")
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=ALL_LAYOUTS,
        metavar="LAYOUT",
        help=f"렌더링할 레이아웃 이름 (선택: {', '.join(ALL_LAYOUTS)}). 미지정 시 전체 실행.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output"),
        help="PNG 저장 디렉터리 (기본값: output/)",
    )
    args = parser.parse_args()

    target_layouts = args.layout if args.layout else ALL_LAYOUTS
    output_dir = Path(args.output_dir)

    print(f"출력 디렉터리: {output_dir}")
    print(f"대상 레이아웃: {target_layouts}\n")

    success, failed = [], []
    for style in target_layouts:
        try:
            out_path = render_layout(style, output_dir)
            print(f"[OK] {style:12s} → {out_path}")
            success.append(style)
        except Exception as exc:
            print(f"[FAIL] {style:12s} → {exc}")
            failed.append(style)

    print(f"\n완료: {len(success)}/{len(target_layouts)} 성공", end="")
    if failed:
        print(f"  실패: {failed}", end="")
    print()


if __name__ == "__main__":
    main()
