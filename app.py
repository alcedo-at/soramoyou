import io
import base64
import hashlib
import numpy as np
import requests
import streamlit as st
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer, util
import pillow_heif

# ==============================
# 初期設定
# ==============================
st.set_page_config(page_title="そらもよう：空を共有して検索", page_icon="🌤️", layout="centered")
pillow_heif.register_heif_opener()

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]  # 例: "alcedo-atthis/soramoyou"
BRANCH = "main"
IMAGE_FOLDER = "images"

DUP_TH = 0.98  # 重複判定閾値（必要なら調整）

# ==============================
# モデルロード
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

model = load_model()

# ==============================
# 画像読み込み（bytes -> PIL）
# ==============================
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        # HEIC fallback
        heif = pillow_heif.read_heif(io.BytesIO(image_bytes))
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
        return img.convert("RGB")

def get_image_feature(image_bytes: bytes) -> np.ndarray:
    img = load_image_from_bytes(image_bytes)
    return model.encode(img, convert_to_numpy=True, normalize_embeddings=True)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ==============================
# GitHub API
# ==============================
def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def list_github_images():
    """
    GitHubの images/ フォルダを列挙
    戻り値: [{"name": str, "download_url": str, "size": int}, ...]
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{IMAGE_FOLDER}?ref={BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json()

    out = []
    for it in items:
        if it.get("type") != "file":
            continue
        name = it.get("name", "")
        ext = name.lower().split(".")[-1] if "." in name else ""
        if ext not in {"jpg", "jpeg", "png", "heic"}:
            continue
        out.append({
            "name": name,
            "download_url": it.get("download_url"),
            "size": int(it.get("size", 0)),
        })
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def download_bytes_cached(url: str) -> bytes:
    """
    GitHubから画像bytesを取得（キャッシュあり）
    """
    r = requests.get(url, headers=gh_headers(), timeout=60)
    r.raise_for_status()
    return r.content

def upload_to_github(image_bytes: bytes, filename: str) -> bool:
    """
    GitHubにファイルを追加（images/filename）
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{IMAGE_FOLDER}/{filename}"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data = {
        "message": f"Add new sky image: {filename}",
        "content": encoded,
        "branch": BRANCH,
    }
    r = requests.put(url, json=data, headers=gh_headers(), timeout=60)
    return r.status_code in (200, 201)

# ==============================
# GitHub画像 -> 特徴量インデックス作成（ローカル保存なし）
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def build_feature_index_from_github():
    """
    GitHub上の images/ を読み込み、特徴量を計算して返す
    戻り値:
      names: List[str]
      urls:  List[str]
      feats: np.ndarray shape=(N, D)
    ※ Streamlitのキャッシュにのみ保持し、ローカルファイルには保存しない。
    """
    items = list_github_images()

    names, urls, feats = [], [], []
    for it in items:
        # 壊れファイル（極小）をスキップ（GitHubのsizeを利用）
        if it["size"] < 1024:
            continue
        if not it["download_url"]:
            continue
        try:
            b = download_bytes_cached(it["download_url"])
            if len(b) < 1024:
                continue
            f = get_image_feature(b)
            names.append(it["name"])
            urls.append(it["download_url"])
            feats.append(f)
        except Exception:
            # 読めないものは除外（ログが欲しければ st.warning にしてもOK）
            continue

    if not feats:
        return [], [], np.array([])
    return names, urls, np.vstack(feats)

def find_top_similar(query_feature: np.ndarray, names, urls, feats, top_k=3):
    sims = util.cos_sim(query_feature, feats)[0].cpu().numpy()
    top_k = min(top_k, len(names))
    idx = sims.argsort()[-top_k:][::-1]
    return [(names[i], urls[i], float(sims[i])) for i in idx], sims

# ==============================
# UI
# ==============================
st.title("🌤️ そらもよう：空を共有して似た空を探す")
st.caption("検索対象は GitHub の images/ のみです（ローカル参照なし）。重複判定後、必要なら保存できます。")

# 既存インデックス準備
with st.spinner("GitHub上の画像を読み込み、特徴量インデックスを準備中..."):
    names, urls, feats = build_feature_index_from_github()

st.write(f"検索対象（GitHub images/）: {len(names)} 枚")

uploaded_file = st.file_uploader(
    "空の写真をアップロード（JPEG/PNG/HEIC対応）",
    type=["jpg", "jpeg", "png", "heic", "HEIC"]
)

if uploaded_file:
    # readは1回だけ
    image_bytes = uploaded_file.read()

    if len(image_bytes) < 1024:
        st.error(f"アップロードされたデータが小さすぎます（{len(image_bytes)} bytes）。アップロード失敗の可能性があります。")
        st.stop()

    # プレビュー（bytesから）
    try:
        preview = load_image_from_bytes(image_bytes)
        st.image(preview, caption="アップロード画像", use_container_width=True)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    # クエリ特徴量
    with st.spinner("アップロード画像の特徴量を抽出中..."):
        query_feature = get_image_feature(image_bytes)

    if len(names) == 0:
        st.warning("比較対象がありません。先にGitHubの images/ に画像を追加してください。")
        st.stop()

    # 類似検索
    with st.spinner("類似画像を検索中..."):
        results, sims = find_top_similar(query_feature, names, urls, feats, top_k=3)

    st.subheader("🌈 類似している空（上位3枚）")
    for name, url, score in results:
        try:
            b = download_bytes_cached(url)           # ★認証つきで取得
            img = load_image_from_bytes(b)           # ★HEICもJPEGもPILへ
            similarity_percent = score * 100
            st.image(img, caption=f"{name} / 類似度:  {similarity_percent:.1f}%", use_container_width=True)
            st.progress(min(score, 1.0))

        except Exception as e:
            # 画像が取れない場合でも「何が起きたか」が分かるようにする
            st.warning(f"{name} の画像表示に失敗しました: {e}")
            st.markdown(f"[画像リンク]({url})")
        # 重複判定
    best_score = float(np.max(sims))
    st.markdown("---")
    st.subheader("📌 この画像を共有（GitHubへ保存）")
    st.write(
    f"重複判定の最大類似度: {best_score*100:.1f}% "
    f"（閾値 {DUP_TH*100:.0f}%）"
    )

    # 追加の完全一致チェック（同一バイト列を弾きたい場合）
    # ※厳密一致だけ弾く用途。必要なければ消してOK。
    upload_hash = sha256_hex(image_bytes)

    if best_score >= DUP_TH:
        st.warning("⚠️ 既存画像と非常に似ているため、重複の可能性が高いです。保存は行いません。")
        st.stop()

    # 保存（最後）
    ext = uploaded_file.name.split(".")[-1].lower()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"

    if st.button("GitHubに保存する"):
        with st.spinner("GitHubに保存中..."):
            ok = upload_to_github(image_bytes, filename)

        if ok:
            st.success("✅ GitHubへの保存が完了しました。")
            raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{IMAGE_FOLDER}/{filename}"
            st.markdown(f"[📸 保存された画像を見る]({raw_url})")
            st.info("※ 検索対象に反映するには、次のリロードでインデックスが更新されます（キャッシュTTL内なら更新されない場合があります）。")
        else:
            st.error("GitHubへのアップロードに失敗しました。トークン権限・Secrets設定・リポジトリ名を確認してください。")

st.markdown("---")
st.caption("SentenceTransformers（clip-ViT-B/32）による埋め込み特徴量を用いて類似検索を実行します。")
