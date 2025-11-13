import os
import io
import glob
import numpy as np
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pillow_heif

# ==============================
# 初期設定
# ==============================
st.set_page_config(page_title="そらもよう：空の類似画像検索", page_icon="☁️", layout="centered")

# HEIC対応（iPhone写真用）
pillow_heif.register_heif_opener()

# モデルロード（CLIP互換の画像対応SentenceTransformer）
@st.cache_resource
def load_model():
    model = SentenceTransformer("clip-ViT-B-32")
    return model

model = load_model()

# ==============================
# 関数定義
# ==============================
def get_image_feature(image_bytes: bytes):
    """画像バイト列から特徴ベクトルを抽出"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # SentenceTransformerのencodeは画像にも対応（CLIPモデルの場合）
    feature = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
    return feature

def find_top_similar_images(query_feature, image_folder="images", top_k=3):
    """指定フォルダ内の画像から類似上位を返す"""
    image_paths = []
    for ext in ["jpg", "jpeg", "png", "heic", "HEIC"]:
        image_paths.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))

    if not image_paths:
        return []

    features = []
    valid_paths = []

    for path in image_paths:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            features.append(get_image_feature(img_bytes))
            valid_paths.append(path)
        except Exception as e:
            st.warning(f"{path} の読み込み中にエラー: {e}")

    if not features:
        return []

    sims = util.cos_sim(query_feature, np.vstack(features))[0].cpu().numpy()
    top_indices = sims.argsort()[-top_k:][::-1]

    return [(valid_paths[i], sims[i]) for i in top_indices]

# ==============================
# Streamlit UI
# ==============================
st.title("☁️ そらもよう：空の類似画像検索")
st.caption("アップロードした空の写真と似ている画像を探します。")

# 検索対象フォルダ
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)

# ファイルアップロード
uploaded_file = st.file_uploader("空の写真をアップロードしてください（JPEG/PNG/HEIC対応）", type=["jpg", "jpeg", "png", "heic", "HEIC"])

if uploaded_file:
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)

    query_bytes = uploaded_file.read()
    query_feature = get_image_feature(query_bytes)

    st.write("🔍 類似画像を検索しています...")
    with st.spinner("検索中..."):
        results = find_top_similar_images(query_feature, image_folder=image_folder, top_k=3)

    if results:
        st.subheader("🌤 類似している空の写真（上位3枚）")
        for path, score in results:
            st.image(path, caption=f"類似度: {score:.3f}", use_container_width=True)
    else:
        st.info("`images/` フォルダに比較対象の画像が見つかりません。")

st.markdown("---")
st.caption("SentenceTransformers（CLIP-ViT-B/32）モデルを使用して空の特徴を比較しています。")
