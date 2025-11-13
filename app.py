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

pillow_heif.register_heif_opener()

# モデルのロード（CLIP互換）
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

model = load_model()

# ディレクトリ設定
IMAGE_FOLDER = "images"
FEATURE_FOLDER = "features"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)

# ==============================
# 関数群
# ==============================
def get_image_feature(image_bytes: bytes):
    """画像バイト列から特徴ベクトルを抽出"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return model.encode(image, convert_to_numpy=True, normalize_embeddings=True)

def build_feature_cache():
    """imagesフォルダの全画像の特徴ベクトルを事前生成・保存"""
    image_paths = []
    for ext in ["jpg", "jpeg", "png", "heic", "HEIC"]:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, f"*.{ext}")))

    st.info(f"検出された画像枚数: {len(image_paths)} 枚")

    for path in image_paths:
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            feature_path = os.path.join(FEATURE_FOLDER, f"{base}.npy")
            if os.path.exists(feature_path):
                continue  # すでに保存済みならスキップ

            with open(path, "rb") as f:
                img_bytes = f.read()
            feature = get_image_feature(img_bytes)
            np.save(feature_path, feature)
        except Exception as e:
            st.warning(f"{path} の特徴量生成中にエラー: {e}")

def load_features():
    """保存済み特徴量を読み込み"""
    features = []
    paths = []
    for npy_path in glob.glob(os.path.join(FEATURE_FOLDER, "*.npy")):
        try:
            features.append(np.load(npy_path))
            base = os.path.splitext(os.path.basename(npy_path))[0]
            # 対応する画像ファイルを探す
            for ext in ["jpg", "jpeg", "png", "heic", "HEIC"]:
                img_path = os.path.join(IMAGE_FOLDER, f"{base}.{ext}")
                if os.path.exists(img_path):
                    paths.append(img_path)
                    break
        except Exception as e:
            st.warning(f"{npy_path} の読み込み中にエラー: {e}")
    return paths, np.vstack(features) if features else np.array([])

def find_top_similar(query_feature, paths, features, top_k=3):
    """類似画像検索"""
    sims = util.cos_sim(query_feature, features)[0].cpu().numpy()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [(paths[i], sims[i]) for i in top_indices]

# ==============================
# Streamlit UI
# ==============================
st.title("☁️ そらもよう：空の類似画像検索")
st.caption("アップロードした空の写真と似ている画像を探します。")

# 特徴ベクトルキャッシュの構築（初回のみ時間がかかる）
with st.spinner("既存画像の特徴ベクトルを確認中..."):
    build_feature_cache()

# アップロードUI
uploaded_file = st.file_uploader("空の写真をアップロードしてください（JPEG/PNG/HEIC対応）", type=["jpg", "jpeg", "png", "heic", "HEIC"])

if uploaded_file:
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
    query_bytes = uploaded_file.read()
    query_feature = get_image_feature(query_bytes)

    st.write("🔍 類似画像を検索しています...")
    with st.spinner("検索中..."):
        paths, features = load_features()
        if len(paths) == 0:
            st.warning("特徴ベクトルが見つかりません。`images/` フォルダに画像を追加してください。")
        else:
            results = find_top_similar(query_feature, paths, features, top_k=3)
            st.subheader("🌤 類似している空の写真（上位3枚）")
            for path, score in results:
                st.image(path, caption=f"類似度: {score:.3f}", use_container_width=True)

st.markdown("---")
st.caption("SentenceTransformers（CLIP-ViT-B/32）モデルを使用。特徴ベクトルをキャッシュし、高速な類似検索を実現しています。")
