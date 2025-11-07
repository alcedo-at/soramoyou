import os
import io
import glob
import numpy as np
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import Part
from sentence_transformers import SentenceTransformer, util

# ==============================
# 設定
# ==============================
st.set_page_config(page_title="そらもようAI", page_icon="🌤️", layout="centered")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("環境変数 GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
    st.stop()

# Geminiクライアント初期化
model_id = "gemini-2.0-flash-exp"
client = genai.Client(api_key=api_key)

# 画像埋め込みモデル（CLIP代替）
st_model = SentenceTransformer("clip-ViT-B-32")

# ==============================
# 関数
# ==============================
def get_image_feature(image_bytes):
    """画像バイト列から特徴ベクトルを抽出"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return st_model.encode(image, convert_to_numpy=True, normalize_embeddings=True)

def find_top_similar_images(query_feature, image_folder="images", top_k=3):
    """指定フォルダ内の画像から類似上位を返す"""
    image_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        image_paths.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))

    if not image_paths:
        return []

    features = []
    for path in image_paths:
        with open(path, "rb") as f:
            img_bytes = f.read()
        features.append(get_image_feature(img_bytes))

    sims = util.cos_sim(query_feature, np.vstack(features))[0].cpu().numpy()
    top_indices = sims.argsort()[-top_k:][::-1]

    return [(image_paths[i], sims[i]) for i in top_indices]

# ==============================
# UI
# ==============================
st.title("🌤️ そらもようAI")
st.caption("空の写真をアップロードすると、日本語表現＋擬態語を生成し、似た空の写真を探します。")

uploaded_file = st.file_uploader("空の写真をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
    st.write("AIが解析中です… 🔍")

    image_bytes = uploaded_file.read()

    # ==============================
    # Geminiによる日本語生成
    # ==============================
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                """次の写真から空を表す日本語と擬態語を出力してください。
絶対に「はい、承知いたしました。」などの応答文は含めないでください。

例：
空を表す日本語：快晴、雲居の空、青嵐、雲の堤
擬態語：ふわふわ、スカッと、もふもふ、すーっ
また、空を表す日本語には簡単な辞書的定義も加えてください。""",
                Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )

        text = getattr(response, "text", None)
        if not text:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = str(response)

        st.subheader("🧭 AIの出力結果")
        st.write(text)

    except Exception as e:
        st.error(f"Gemini呼び出し中にエラーが発生しました：{e}")
        st.stop()

    # ==============================
    # 類似画像検索
    # ==============================
    st.subheader("🔍 類似する空の写真を検索")
    query_feature = get_image_feature(image_bytes)

    with st.spinner("類似画像を検索中..."):
        results = find_top_similar_images(query_feature)

    if results:
        for path, score in results:
            st.image(path, caption=f"類似度：{score:.3f}", use_container_width=True)
    else:
        st.info("`images/` フォルダに比較対象の画像が見つかりません。")

st.markdown("---")
st.caption("Gemini + CLIP代替モデル（SentenceTransformers）を使用しています。")
