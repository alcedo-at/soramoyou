import streamlit as st
from google import genai
from google.genai.types import Part
from dotenv import load_dotenv
import os

# ======== 設定 ========
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("環境変数 GEMINI_API_KEY が設定されていません。`.env` ファイルを確認してください。")
    st.stop()

# ✅ モデルIDを修正
model_id = "gemini-2.0-flash-exp"

# クライアント初期化
client = genai.Client(api_key=api_key)

# ======== Streamlit UI ========
st.set_page_config(page_title="そらもようAI", page_icon="🌤️", layout="centered")
st.title("🌤️ そらもようAI")
st.caption("画像をアップロードすると、空を表す日本語と擬態語をAIが提案します。")

uploaded_file = st.file_uploader("空の写真をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
    st.write("AIが画像を分析しています… 🔍")

    image_bytes = uploaded_file.read()

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                """次の写真から空を表す日本語と擬態語を出力してください。
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

        st.subheader("🧭 AIの出力結果")
        st.write(text)

    except Exception as e:
        st.error(f"API呼び出し中にエラーが発生しました：{e}")
