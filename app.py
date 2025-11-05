import os
import glob
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.types import Part
from dotenv import load_dotenv
import pillow_heif
import streamlit as st

# ======== 初期設定 ========
pillow_heif.register_heif_opener()
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("環境変数 GEMINI_API_KEY が設定されていません。`.env` ファイルを確認してください。")
    st.stop()

client = genai.Client(api_key=api_key)
model_id = "gemini-2.0-flash-exp"

# CLIPモデルの準備
def load_clip_model():
    device = "cpu"  # Cloud上では常にCPU
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

# ======== 関数定義 ========

def describe_image_with_gemini(client, image_bytes):
    """Geminiで空を表す日本語と擬態語を生成"""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                """次の写真から空を表す日本語と擬態語を出力してください。
                 絶対に「はい、承知いたしました」などの定型文は返さないでください。
例：
空を表す日本語：快晴、雲居の空、青嵐、雲の堤  
擬態語：ふわふわ、スカッと、もふもふ、すーっ  

また、空を表す日本語は辞書的な定義も出力してください。""",
                Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
        )
        text = getattr(response, "text", None)
        if not text:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = str(response)
        return text
    except Exception as e:
        return f"API呼び出し中にエラーが発生しました：{e}"

def get_image_feature(image_path_or_file):
    """CLIPで画像特徴ベクトルを抽出（HEIC対応・正規化付き）"""
    if isinstance(image_path_or_file, str):
        image = Image.open(image_path_or_file)
    else:
        image = Image.open(image_path_or_file)
    image = image.convert("RGB").resize((512, 512))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = clip_model.encode_image(image_input)
    feature /= feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy()[0]

def find_top_similar(query_feature, features_dict, top_k=3):
    """辞書内の画像ベクトルから類似度上位K件を検索"""
    paths = list(features_dict.keys())
    features = np.array(list(features_dict.values()))
    sims = cosine_similarity([query_feature], features)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(paths[i], sims[i]) for i in top_indices]
    return results

# ======== Streamlit UI ========
st.set_page_config(page_title="そらもようAI", page_icon="🌤️", layout="centered")
st.title("🌤️ そらもようAI")
st.caption("画像をアップロードすると、空を表す日本語と擬態語をAIが提案し、似た空の写真を表示します。")

# 類似画像フォルダ
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
              glob.glob(os.path.join(image_folder, "*.jpeg")) + \
              glob.glob(os.path.join(image_folder, "*.png")) + \
              glob.glob(os.path.join(image_folder, "*.heic"))

# 既存画像の特徴量を事前計算
features_dict = {}
if image_paths:
    with st.spinner("既存の画像を解析中..."):
        for path in image_paths:
            try:
                features_dict[path] = get_image_feature(path)
            except Exception as e:
                st.warning(f"{path} の読み込み中にエラー: {e}")

uploaded_file = st.file_uploader("空の写真をアップロードしてください（JPEG/PNG/HEIC）", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file:
    clip_model, preprocess = load_clip_model()
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
    image_bytes = uploaded_file.read()

    # --- Geminiによる解析 ---
    with st.spinner("AIが空を分析しています… 🔍"):
        result_text = describe_image_with_gemini(client, image_bytes)

    st.subheader("🧭 AIの出力結果")
    st.write(result_text)

    # --- CLIPによる類似検索 ---
    if features_dict:
        query_feature = get_image_feature(uploaded_file)
        with st.spinner("類似画像を検索中..."):
            top_results = find_top_similar(query_feature, features_dict, top_k=3)

        st.subheader("🔍 類似している空の写真（上位3枚）")
        cols = st.columns(3)
        for col, (path, score) in zip(cols, top_results):
            col.image(path, caption=f"類似度：{score:.3f}", use_container_width=True)
    else:
        st.info("比較対象となる画像がフォルダ内にありません。`images/` フォルダに空の写真を追加してください。")

st.markdown("---")
st.caption("Gemini API + CLIPにより、空の日本語表現と類似空模様の検索を行います。")
