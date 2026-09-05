"""Streamlit UI for the floor-plan analyzer."""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from floor_plan import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, analyze_floor_plan, create_analysis_client, load_knowledge


APP_DIR = PROJECT_DIR
KNOWLEDGE_PATH = APP_DIR / "knowledge.md"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

st.set_page_config(page_title="AI 間取り分析", page_icon="🏠", layout="wide")
load_dotenv(APP_DIR / ".env")


@st.cache_data
def get_knowledge(path: str, modified_ns: int) -> str:
    """Cache knowledge until the file's modification time changes."""
    del modified_ns
    return load_knowledge(path)


@st.cache_resource
def get_analysis_client(provider: str, host: str, num_ctx: int, max_images: int, api_key: str, fallback: bool):
    return create_analysis_client(
        provider,
        ollama_host=host,
        ollama_num_ctx=num_ctx,
        ollama_max_images=max_images,
        gemini_api_key=api_key,
        enable_gemini_fallback=fallback,
    )


def open_uploaded_image(uploaded_file) -> Image.Image:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("画像サイズは15MB以下にしてください。")

    image = Image.open(uploaded_file)
    image.load()
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


st.title("🏠 AI 間取り分析")
st.caption("knowledge.mdの全項目と固定の100点採点表を使って評価します。")

api_key = os.getenv("GEMINI_API_KEY", "").strip()
provider = os.getenv("ANALYSIS_PROVIDER", "ollama").strip().lower()
default_model = DEFAULT_MODEL if provider == "ollama" else "gemini-2.5-flash"
model = os.getenv("OLLAMA_MODEL" if provider == "ollama" else "GEMINI_MODEL", default_model).strip() or default_model
fallback = os.getenv("ENABLE_GEMINI_FALLBACK", "false").strip().lower() in {"1", "true", "yes"}
ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
ollama_max_images = int(os.getenv("OLLAMA_MAX_IMAGES", "1"))

try:
    client = get_analysis_client(provider, ollama_host, ollama_num_ctx, ollama_max_images, api_key, fallback)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    knowledge = get_knowledge(str(KNOWLEDGE_PATH), KNOWLEDGE_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "間取り画像を選択",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    try:
        image = open_uploaded_image(uploaded_file)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"画像を読み込めません: {exc}")
        st.stop()

    image_column, result_column = st.columns([1, 1.2])
    with image_column:
        st.image(image, caption="アップロード画像", use_container_width=True)
        analyze_button = st.button("🔍 2段階で分析する", type="primary")

    with result_column:
        if analyze_button:
            with st.spinner("空間の接続関係を構造化してから採点中..."):
                try:
                    result = analyze_floor_plan(
                        image=image,
                        client=client,
                        knowledge=knowledge,
                        model=model,
                    )
                except Exception as exc:
                    st.error(f"分析に失敗しました: {exc}")
                else:
                    st.success("分析が完了しました。")
                    with st.expander("第1段階：最初の読取結果"):
                        st.json(result.draft_structure)
                    with st.expander("第1段階：再照合後の読取結果", expanded=True):
                        st.json(result.structure)
                    with st.expander("第2段階：検証済み採点JSON", expanded=True):
                        st.json(result.scoring)
                    with st.container(border=True):
                        st.markdown(result.report)
