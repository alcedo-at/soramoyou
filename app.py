import base64
import glob
import hashlib
import io
import math
import random
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat, UnidentifiedImageError, ImageOps
from sentence_transformers import util

import clip
import pillow_heif
from google import genai
from google.genai.types import Part
from sklearn.metrics.pairwise import cosine_similarity
import threading
import json

# ============================================================
# そらもよう：空を共有して似た空を探す
# - 類似画像検索（index.npz + cosine similarity）は既存ロジックを極力維持
# - 画像への単語描画は、ボタン押下時のみ実行し session_state で保持
# - Gemini は任意（ボタン押下時のみ / sha256 単位でキャッシュ）
# ============================================================

# ------------------------------
# Streamlit / HEIC
# ------------------------------
st.set_page_config(page_title="そらもよう：空を共有して検索", page_icon="🌤️", layout="centered")
pillow_heif.register_heif_opener()

# ------------------------------
# Secrets
# ------------------------------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GITHUB_REPO = st.secrets.get("GITHUB_REPO")  # "user/repo"
BRANCH = st.secrets.get("BRANCH", "main")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = st.secrets.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

IMAGE_FOLDER = "images"
FEATURES_INDEX_PATH = "features/index.npz"

TOP_K = 3
DUP_TH = 0.98  # 類似度がこれ以上なら重複扱い

SMALL_SIZE_TH = 1024  # bytes（壊れ画像っぽい極小データの閾値）

# ------------------------------
# Early validation
# ------------------------------
if not GITHUB_TOKEN or not GITHUB_REPO:
    st.error(
        "Secrets に GITHUB_TOKEN / GITHUB_REPO が設定されていません。"
        "Streamlit Cloud の Manage app → Secrets を確認してください。"
    )
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
CLIP_INFER_LOCK = threading.Lock()

# ------------------------------
# Model (CLIP)
# ------------------------------
@st.cache_resource
def load_clip_model():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return device, model, preprocess


DEVICE, CLIP_MODEL, PREPROCESS = load_clip_model()

# ============================================================
# Utilities
# ============================================================
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """画像bytesからPIL画像を生成します（EXIF回転を補正し、HEICも対応）。"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # 縦写真が横になる問題を抑止
        return img.convert("RGB")
    except UnidentifiedImageError:
        heif = pillow_heif.read_heif(io.BytesIO(image_bytes))
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
        # HEICはEXIF回転が別管理の場合があるため、念のため適用
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def get_image_feature(image_path_or_file):
    """
    CLIPで画像特徴ベクトルを抽出（HEIC対応・正規化付き / 入力柔軟版）
    - str: ローカルパス
    - bytes: 画像バイト列
    - PIL.Image.Image: 画像
    - file-like: Streamlit UploadedFile 等
    """
    if isinstance(image_path_or_file, str):
        image = ImageOps.exif_transpose(Image.open(image_path_or_file))
    elif isinstance(image_path_or_file, bytes):
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_path_or_file)))
    elif isinstance(image_path_or_file, Image.Image):
        image = image_path_or_file
    else:
        image = ImageOps.exif_transpose(Image.open(image_path_or_file))

    image = image.convert("RGB").resize((512, 512))
    image_input = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with CLIP_INFER_LOCK:
        with torch.no_grad():
            feature = CLIP_MODEL.encode_image(image_input)

    feature = feature / feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy()[0]


def get_text_feature(word: str) -> np.ndarray:
    """CLIPのtext encoderで単語埋め込みを作り、正規化して返す"""
    text = clip.tokenize([word]).to(DEVICE)
    with CLIP_INFER_LOCK:
        with torch.no_grad():
            feat = CLIP_MODEL.encode_text(text)

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


# ============================================================
# GitHub API helpers
# ============================================================
def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def gh_contents_url(path: str) -> str:
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"


def gh_get_content_json(path: str):
    """GitHub contents API GET。存在しないなら None。"""
    r = requests.get(gh_contents_url(path), headers=gh_headers(), params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=3600)
def gh_download_bytes_cached(url: str, version: str = "") -> bytes:
    """download_url から bytes を取得（キャッシュ）。version でキャッシュ無効化できる。"""
    _ = version  # cache key
    r = requests.get(url, headers=gh_headers(), timeout=60)
    r.raise_for_status()
    return r.content


def gh_put_file(path: str, content_bytes: bytes, message: str, sha: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """成功したら (True, new_sha)、失敗したら (False, None)"""
    url = gh_contents_url(path)
    encoded = base64.b64encode(content_bytes).decode("utf-8")
    payload = {"message": message, "content": encoded, "branch": BRANCH}
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=gh_headers(), json=payload, timeout=60)
    if r.status_code not in (200, 201):
        return False, None

    js = r.json()
    new_sha = None
    try:
        new_sha = js["content"]["sha"]
    except Exception:
        pass
    return True, new_sha


def gh_delete_file(path: str, sha: str, message: str) -> bool:
    """
    GitHub contents API でファイル削除（DELETE）。
    path: "images/xxx.jpg" など
    sha : contents API で取得できるそのファイルの sha
    """
    url = gh_contents_url(path)
    payload = {"message": message, "sha": sha, "branch": BRANCH}
    r = requests.delete(url, headers=gh_headers(), json=payload, timeout=60)
    return r.status_code == 200


def list_github_images() -> List[Dict]:
    """GitHub images/ を列挙して file item を返す。"""
    data = gh_get_content_json(IMAGE_FOLDER)
    if not isinstance(data, list):
        return []
    out: List[Dict] = []
    for it in data:
        if it.get("type") != "file":
            continue
        name = it.get("name", "")
        ext = name.lower().split(".")[-1] if "." in name else ""
        if ext not in {"jpg", "jpeg", "png", "heic"}:
            continue
        out.append(
            {
                "name": name,
                "download_url": it.get("download_url"),
                "size": int(it.get("size", 0)),
            }
        )
    return out


# ============================================================
# index.npz pack/unpack
# ============================================================
def index_pack(
    names: List[str],
    sha_list: List[str],
    embeddings: np.ndarray,
    comments: Optional[List[str]] = None,
) -> bytes:
    """
    names: 画像ファイル名
    sha_list: 画像bytesのsha256（重複防止用）
    embeddings: shape=(N, D)
    comments: 画像に紐づくコメント（任意、N要素）
    """
    if comments is None:
        comments = [""] * len(names)
    # 不整合を安全側で補正（短ければ埋め、長ければ切る）
    if len(comments) != len(names):
        if len(comments) < len(names):
            comments = comments + [""] * (len(names) - len(comments))
        else:
            comments = comments[: len(names)]

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        names=np.array(names, dtype=np.str_),
        sha256=np.array(sha_list, dtype=np.str_),
        embeddings=embeddings.astype(np.float32, copy=False),
        comments=np.array(comments, dtype=np.str_),
    )
    return buf.getvalue()


def index_unpack(b: bytes) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """index.npz bytes -> (names, sha256, embeddings, comments)"""
    z = np.load(io.BytesIO(b), allow_pickle=False)
    names = z["names"].astype(str).tolist()
    sha_list = z["sha256"].astype(str).tolist()
    embeddings = z["embeddings"].astype(np.float32, copy=False)
    if "comments" in z.files:
        comments = z["comments"].astype(str).tolist()
    else:
        comments = [""] * len(names)

    if len(comments) != len(names):
        if len(comments) < len(names):
            comments = comments + [""] * (len(names) - len(comments))
        else:
            comments = comments[: len(names)]

    return names, sha_list, embeddings, comments


def load_index_from_github() -> Tuple[List[str], List[str], np.ndarray, List[str], Optional[str]]:
    """
    GitHub の features/index.npz を読み込み。
    戻り値: (names, sha256, embeddings, comments, index_sha) または ([], [], empty, [], None)
    """
    meta = gh_get_content_json(FEATURES_INDEX_PATH)
    if not meta or not meta.get("download_url"):
        return [], [], np.array([], dtype=np.float32), [], None

    b = gh_download_bytes_cached(meta["download_url"], version=meta.get("sha", ""))
    names, sha_list, embeddings, comments = index_unpack(b)
    return names, sha_list, embeddings, comments, meta.get("sha")


def sync_index_missing_images(
    image_items: List[Dict],
    idx_names: List[str],
    idx_sha_list: List[str],
    idx_embeddings: np.ndarray,
    idx_file_sha: Optional[str],
    idx_comments: List[str],
) -> Tuple[List[str], List[str], np.ndarray, List[str], Optional[str], int, bool]:
    """
    images/ にあるのに index.npz に無い画像だけを計算して index.npz を更新する。
    """
    image_name_to_url = {it["name"]: it.get("download_url") for it in image_items}
    images_set = set(image_name_to_url.keys())
    index_set = set(idx_names)

    missing = sorted(images_set - index_set)
    if not missing:
        return idx_names, idx_sha_list, idx_embeddings, idx_comments, idx_file_sha, 0, True

    new_names: List[str] = []
    new_shas: List[str] = []
    new_vecs: List[np.ndarray] = []

    for name in missing:
        url = image_name_to_url.get(name)
        if not url:
            continue
        try:
            img_bytes = gh_download_bytes_cached(url)
            if len(img_bytes) < SMALL_SIZE_TH:
                continue
            vec = get_image_feature(img_bytes)
            new_names.append(name)
            new_shas.append(sha256_hex(img_bytes))
            new_vecs.append(vec)
        except Exception:
            continue

    if not new_vecs:
        return idx_names, idx_sha_list, idx_embeddings, idx_comments, idx_file_sha, len(missing), False

    emb2 = np.vstack([idx_embeddings, np.vstack(new_vecs)]).astype(np.float32, copy=False) if len(idx_embeddings) else np.vstack(new_vecs).astype(np.float32, copy=False)
    names2 = idx_names + new_names
    sha2 = idx_sha_list + new_shas
    comments2 = idx_comments + [""] * len(new_names)
    payload = index_pack(names2, sha2, emb2, comments=comments2)

    ok, _ = gh_put_file(
        FEATURES_INDEX_PATH,
        payload,
        message=f"Sync index.npz (+{len(new_names)} images)",
        sha=idx_file_sha,
    )
    return names2, sha2, emb2, comments2, idx_file_sha, len(new_names), ok


def build_index_from_images_and_upload() -> Tuple[int, bool]:
    """
    images/ を走査して index.npz を生成し、GitHubに保存。
    """
    items = list_github_images()
    if len(items) == 0:
        return 0, False

    names: List[str] = []
    sha_list: List[str] = []
    embs: List[np.ndarray] = []

    prog = st.progress(0.0)
    done = 0
    total = len(items)

    for it in items:
        done += 1
        prog.progress(done / total)

        if it.get("size", 0) < SMALL_SIZE_TH:
            continue
        url = it.get("download_url")
        if not url:
            continue
        try:
            img_bytes = gh_download_bytes_cached(url)
            if len(img_bytes) < SMALL_SIZE_TH:
                continue
            vec = get_image_feature(img_bytes)
            names.append(it["name"])
            sha_list.append(sha256_hex(img_bytes))
            embs.append(vec)
        except Exception:
            continue

    if len(embs) == 0:
        return 0, False

    embeddings = np.vstack(embs).astype(np.float32, copy=False)
    comments = [""] * len(names)
    payload = index_pack(names, sha_list, embeddings, comments=comments)
    ok, _ = gh_put_file(FEATURES_INDEX_PATH, payload, message="Build features index (index.npz)", sha=None)
    return len(names), ok


def append_to_index_and_upload(new_name: str, new_sha: str, new_vec: np.ndarray, new_comment: str = "") -> bool:
    """
    index.npz に1件追記して GitHub へ更新。
    競合（sha不一致）の可能性があるので、失敗時は1回だけ再取得して再試行します。
    """
    names, sha_list, embeddings, comments, index_sha = load_index_from_github()
    if len(names) == 0 and index_sha is None:
        payload = index_pack([new_name], [new_sha], new_vec.reshape(1, -1), comments=[new_comment])
        ok, _ = gh_put_file(
            FEATURES_INDEX_PATH,
            payload,
            message=f"Create features index with {new_name}",
            sha=None,
        )
        return ok

    if new_sha in sha_list:
        return True

    names2 = names + [new_name]
    sha2 = sha_list + [new_sha]
    emb2 = np.vstack([embeddings, new_vec.reshape(1, -1)]).astype(np.float32, copy=False)
    comments2 = comments + [new_comment]
    payload2 = index_pack(names2, sha2, emb2, comments=comments2)

    ok, _ = gh_put_file(
        FEATURES_INDEX_PATH,
        payload2,
        message=f"Update features index (+{new_name})",
        sha=index_sha,
    )
    if ok:
        return True

    names, sha_list, embeddings, comments, index_sha = load_index_from_github()
    if index_sha is None:
        return False
    if new_sha in sha_list:
        return True

    names2 = names + [new_name]
    sha2 = sha_list + [new_sha]
    emb2 = np.vstack([embeddings, new_vec.reshape(1, -1)]).astype(np.float32, copy=False)
    comments2 = comments + [new_comment]
    payload2 = index_pack(names2, sha2, emb2, comments=comments2)

    ok, _ = gh_put_file(
        FEATURES_INDEX_PATH,
        payload2,
        message=f"Update features index (+{new_name})",
        sha=index_sha,
    )
    return ok


# ============================================================
# Similar search (keep behavior)
# ============================================================
def find_top_similar(query_vec: np.ndarray, names: List[str], embeddings: np.ndarray, top_k: int):
    sims = util.cos_sim(query_vec, embeddings)[0].cpu().numpy()
    k = min(top_k, len(names))
    idx = sims.argsort()[-k:][::-1]
    return [(names[i], float(sims[i])) for i in idx], sims


def image_download_url_by_name(name: str, items: List[Dict]) -> Optional[str]:
    for it in items:
        if it["name"] == name:
            return it.get("download_url")
    return None


# ============================================================
# Gemini word generation
# ============================================================
def extract_words_from_text(result_text: str, max_words: int = 10) -> List[str]:
    """
    Geminiの自由文から、候補単語をざっくり抽出（暫定）。
    「：」以降を優先して「、」「,」「改行」で分割。
    """
    if not result_text:
        return []

    candidates: List[str] = []

    for line in result_text.splitlines():
        line = line.strip()
        if "：" in line:
            head, tail = line.split("：", 1)
            if "とは" in tail or "定義" in head:
                continue
            parts = re.split(r"[、,]\s*", tail)
            candidates.extend([p.strip() for p in parts if p.strip()])

    if not candidates:
        tokens = re.findall(r"[一-龠ぁ-んァ-ヶー]{2,}", result_text)
        candidates = tokens

    seen = set()
    uniq: List[str] = []
    for w in candidates:
        if w not in seen:
            seen.add(w)
            uniq.append(w)

    return uniq[:max_words]


def _strip_code_fences(s: str) -> str:
    """Geminiが ```json ... ``` で返した場合に備えてフェンスを除去します。"""
    if not s:
        return s
    s2 = s.strip()
    if s2.startswith("```"):
        # 先頭行の ```json / ``` を除去
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s2)
        # 末尾の ``` を除去
        s2 = re.sub(r"\s*```\s*$", "", s2)
    return s2.strip()


def gemini_generate_words(image_bytes: bytes) -> Tuple[List[str], Dict[str, str], str]:
    """
    Geminiで単語を生成し、
      - words: CLIP評価に使う「単語だけ」のリスト
      - defs : {単語: 辞書的な定義}（表示用）
      - raw_text: Geminiの生出力
    を返します。

    失敗した場合は ([], {}, error_message)。
    """
    if gemini_client is None:
        return [], {}, "GEMINI_API_KEY が設定されていません。"

    # 重要: CLIP一致度に解説文を混ぜないため、Geminiの出力を JSON のみに固定します。
    # 「辞書的な定義」は defs として保持し、表示だけに使います。
    prompt = (
        "次の条件を厳守して、JSON だけを出力してください。前置き、注釈、箇条書き、見出し、コードブロックは一切不要です。"
        "\n\n"
        "【出力形式】\n"
        "{\n"
        '  "sky_words": [\n'
        '    {"word": "単語", "definition": "辞書的な定義（短く）"}, ...\n'
        "  ],\n"
        '  "onomatopoeia": [\n'
        '    {"word": "擬態語", "definition": "意味（短く）"}, ...\n'
        "  ]\n"
        "}\n\n"
        "【内容条件】\n"
        "- sky_words は空を表す日本語（最大4個）\n"
        "- onomatopoeia は擬態語（最大4個）\n"
        "- 同じ語をひらがな・漢字・カタカナで重複させない\n"
        "- definition は短く、辞書的に（1文程度）\n"
    )

    try:
        part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[prompt, part],
        )

        raw_text = resp.text if hasattr(resp, "text") and resp.text else str(resp)
        cleaned = _strip_code_fences(raw_text)

        words: List[str] = []
        defs: Dict[str, str] = {}

        try:
            data = json.loads(cleaned)
            for key in ("sky_words", "onomatopoeia"):
                arr = data.get(key, [])
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    w = str(it.get("word", "")).strip()
                    d = str(it.get("definition", "")).strip()
                    if not w:
                        continue
                    if w not in defs:
                        words.append(w)
                        defs[w] = d
        except Exception:
            # JSONが崩れて返る場合に備え、従来の抽出ロジックにフォールバック（定義は空）
            words = extract_words_from_text(raw_text, max_words=10)
            defs = {w: "" for w in words}

        return words, defs, raw_text

    except Exception as e:
        return [], {}, f"Gemini呼び出しに失敗しました: {e}"


# ============================================================
# Word scoring + overlay (variable size)
# ============================================================
def score_words_by_clip(image_feature: np.ndarray, words: Sequence[str]) -> List[Tuple[str, float]]:
    """
    画像特徴ベクトル（正規化済み）と単語特徴ベクトルのコサイン類似度を返す
    - score は 0〜1 に正規化した値
    """
    words = [w.strip() for w in words if w and w.strip()]
    if len(words) == 0:
        return []

    text_feats = np.vstack([get_text_feature(w) for w in words])  # (N, D)
    sims = cosine_similarity([image_feature], text_feats)[0]  # (N,)

    sims01 = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    pairs = list(zip(words, sims01.tolist()))
    # 高スコア順
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def font_size_from_score(
    score01: float,
    min_size: int = 18,
    max_size: int = 60,
    gamma: float = 2.2,
) -> int:
    """0..1 のスコアをフォントサイズへ変換します。

    gamma > 1 で高スコアを強調し、低スコアを小さくします（差が出ます）。
    """
    s = float(np.clip(score01, 0.0, 1.0))
    g = float(max(0.1, gamma))
    s = s**g
    return int(round(min_size + (max_size - min_size) * s))


def choose_text_color(image: Image.Image) -> Tuple[int, int, int]:
    gray = image.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)


def find_smooth_region(image: Image.Image, window: int = 120) -> Tuple[int, int]:
    """
    エッジ（輪郭）量が少ない領域の中心を返す。
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float32)
    h, w = arr.shape[:2]
    window = int(np.clip(window, 40, min(h, w)))

    step = max(1, window // 2)

    best: Optional[Tuple[int, int]] = None
    best_var: Optional[float] = None

    for y in range(0, max(1, h - window + 1), step):
        for x in range(0, max(1, w - window + 1), step):
            block = arr[y : y + window, x : x + window]
            v = float(block.var())
            if best_var is None or v < best_var:
                best_var = v
                best = (x + window // 2, y + window // 2)

    if best is None:
        return (image.width // 2, image.height // 2)
    return best

def find_smooth_regions(
    image: Image.Image,
    window: int = 120,
    top_k: int = 12,
) -> List[Tuple[int, int]]:
    """エッジ量が少ない領域（候補）を複数返す。

    画像全体を粗いグリッドで走査し、ブロック内のエッジ画像の分散が小さい順に上位を返します。
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float32)
    h, w = arr.shape[:2]
    window = int(np.clip(window, 40, min(h, w)))

    step = max(1, window // 2)

    candidates: List[Tuple[float, int, int]] = []
    for y in range(0, max(1, h - window + 1), step):
        for x in range(0, max(1, w - window + 1), step):
            block = arr[y : y + window, x : x + window]
            v = float(block.var())
            candidates.append((v, x + window // 2, y + window // 2))

    candidates.sort(key=lambda t: t[0])
    centers: List[Tuple[int, int]] = []
    for v, cx, cy in candidates[: max(1, int(top_k))]:
        centers.append((int(cx), int(cy)))

    if not centers:
        centers = [(image.width // 2, image.height // 2)]
    return centers



def overlay_words_variable_size(
    image_bytes: bytes,
    word_scores: Sequence[Tuple[str, float]],
    *,
    font_path: str = "fonts/NotoSansJP-Bold.ttf",
    padding_ratio: float = 0.05,
    min_font: int = 0,
    max_font: int = 0,
    size_gamma: float = 2.2,
    size_contrast: float = 2.0,
    scatter_strength: float = 0.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    word_scores: [(word, score01), ...]

    - score に応じてフォントサイズを変える
    - フォントサイズは、原則として画像サイズ（短辺）に比例して決める
      - min_font / max_font が 1 以上なら、その値を優先（固定サイズを使いたい場合）
      - 0 以下なら、画像サイズから自動決定
    - 配置は scatter_strength に応じて「縦積み」または「散布」
    """
    img = load_image_from_bytes(image_bytes)
    draw = ImageDraw.Draw(img)

    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"フォントが見つかりません: {font_path}\n"
            "fonts/ に日本語フォント（.ttf）を置き、font_path を合わせてください。"
        )

    W, H = img.size
    short_side = min(W, H)

    # ------------------------------
    # 画像サイズに応じたフォントサイズ（核心）
    # ------------------------------
    # UIから固定値を渡す場合に備え、1以上は固定値として採用。
    # 0以下は画像サイズから推定。
    if int(min_font) > 0:
        eff_min_font = int(min_font)
    else:
        eff_min_font = max(14, int(round(short_side * 0.035)))  # 3.5%

    if int(max_font) > 0:
        eff_max_font = int(max_font)
    else:
        eff_max_font = max(eff_min_font + 2, int(round(short_side * 0.070)))  # 7.0%

    pad = max(8, int(W * float(padding_ratio)))
    max_w = W - 2 * pad
    max_h = H - 2 * pad

    fill = choose_text_color(img)
    cx, cy = find_smooth_region(img, window=min(160, max(80, W // 6)))

    # スコアが近いとフォントサイズ差が出にくいので、単語集合内で相対正規化してコントラストを付与
    scores = [float(s) for _, s in word_scores] if word_scores else []
    mn = min(scores) if scores else 0.0
    mx = max(scores) if scores else 1.0
    spread = mx - mn
    contrast = float(max(0.1, size_contrast))

    def _adjust_score(s: float) -> float:
        s = float(np.clip(s, 0.0, 1.0))
        if spread > 1e-6:
            s = (s - mn) / spread  # 0..1（相対）
        # コントラスト：中心0.5を基準に伸縮
        s = (s - 0.5) * contrast + 0.5
        return float(np.clip(s, 0.0, 1.0))

    items: List[List] = []
    for word, s01 in word_scores:
        s_adj = _adjust_score(float(s01))
        fs = font_size_from_score(
            s_adj,
            min_size=eff_min_font,
            max_size=eff_max_font,
            gamma=size_gamma,
        )
        font = ImageFont.truetype(font_path, fs)
        bbox = draw.textbbox((0, 0), word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        items.append([word, float(s01), fs, font, tw, th])

    gap = 10

    # まず、各単語が余白内に収まる程度までフォントを調整（保険）
    for _ in range(12):
        if not items:
            break
        too_big = False
        for it in items:
            if it[4] > max_w or it[5] > max_h:
                too_big = True
                it[2] = max(14, int(it[2] * 0.9))  # fs
                it[3] = ImageFont.truetype(font_path, it[2])
                bbox = draw.textbbox((0, 0), it[0], font=it[3])
                it[4], it[5] = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if not too_big:
            break

    scatter = float(np.clip(scatter_strength, 0.0, 1.0))

    # ------------------------------
    # 縦積み（従来）
    # ------------------------------
    if scatter <= 0.0:
        block_w = max(it[4] for it in items) if items else 0
        block_h = sum(it[5] for it in items) + gap * (len(items) - 1 if len(items) >= 2 else 0)

        x0 = int(cx - block_w / 2)
        y0 = int(cy - block_h / 2)

        x0 = max(pad, min(x0, W - pad - block_w))
        y0 = max(pad, min(y0, H - pad - block_h))

        y = y0
        for word, s01, fs, font, tw, th in items:
            x = x0 + (block_w - tw) / 2
            draw.text((x, y), word, font=font, fill=fill)
            y += th + gap

        return img

    # ------------------------------
    # 散布配置（重なり回避）
    # ------------------------------
    if seed is None:
        seed = int(hashlib.sha256(str(word_scores).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    centers = find_smooth_regions(
        img,
        window=min(200, max(90, W // 5)),
        top_k=min(30, max(8, len(items) * 3)),
    )

    radius = int(short_side * (0.08 + 0.28 * scatter))
    placed: List[Tuple[int, int, int, int]] = []  # x1,y1,x2,y2

    def intersects(r1, r2) -> bool:
        return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])

    # 大きい単語から先に置く（重なりにくい）
    items_sorted = sorted(items, key=lambda it: it[2], reverse=True)

    for idx_it, it in enumerate(items_sorted):
        word, s01, fs, font, tw, th = it
        ok = False

        tries = min(80, 20 + len(centers) * 4)
        for t in range(tries):
            base = centers[(idx_it + t) % len(centers)]
            angle = rng.random() * 2.0 * math.pi
            r = radius * (0.25 + 0.75 * rng.random())
            ox = int(math.cos(angle) * r)
            oy = int(math.sin(angle) * r)

            cx2 = base[0] + ox
            cy2 = base[1] + oy

            x = int(cx2 - tw / 2)
            y = int(cy2 - th / 2)

            # 余白内へクランプ
            x = max(pad, min(x, W - pad - tw))
            y = max(pad, min(y, H - pad - th))

            rect = (x - 4, y - 4, x + tw + 4, y + th + 4)

            if any(intersects(rect, r2) for r2 in placed):
                continue

            draw.text((x, y), word, font=font, fill=fill)
            placed.append(rect)
            ok = True
            break

        if not ok:
            # どうしても置けない場合はフォールバック（重なり許容）
            x = int(cx - tw / 2)
            y = int(cy - th / 2)
            x = max(pad, min(x, W - pad - tw))
            y = max(pad, min(y, H - pad - th))
            draw.text((x, y), word, font=font, fill=fill)
            placed.append((x - 4, y - 4, x + tw + 4, y + th + 4))

    return img


def overlay_cache_key(image_sha: str, words: Sequence[str]) -> str:
    joined = "\n".join([w.strip() for w in words if w and w.strip()])
    wsha = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{image_sha}:{wsha}"


# ============================================================
# UI
# ============================================================
st.markdown("---")
st.title("🌤️ そらもよう～空をことばで感じるAIシステムの開発～")
# ============================================================
# 研究の目的（Webページ上部に表示）
# ============================================================
st.markdown(
"""
「そらもよう」は日本語特有の擬態語・情緒表現を通して、気象にもっと気軽に興味を持ってもらうために生まれた、
空の写真から直感的な日本語表現を生成するAIシステムです。
"""
)

# session_state init
st.session_state.setdefault("gemini_cache", {})  # sha256(image_bytes) -> {"words":[...], "defs":{word:def}, "raw":"..."}
st.session_state.setdefault("overlay_cache", {})  # overlay_cache_key -> PIL.Image.Image

with st.spinner("GitHub images/ を読み込み中..."):
    image_items = list_github_images()
st.write(f"GitHub images/ の画像数: {len(image_items)} 枚")

with st.spinner("features/index.npz を確認中..."):
    idx_names, idx_sha_list, idx_embeddings, idx_comments, idx_file_sha = load_index_from_github()

# index が無い場合は作る
if idx_file_sha is None:
    st.warning("features/index.npz が見つかりません。images/ から生成します。")
    with st.spinner("index.npz を生成中..."):
        n, ok = build_index_from_images_and_upload()
    if ok:
        st.success(f"index.npz を生成しました（{n} 枚）。")
        idx_names, idx_sha_list, idx_embeddings, idx_comments, idx_file_sha = load_index_from_github()
    else:
        st.error("index.npz の生成に失敗しました。images/ の状態や権限を確認してください。")
        st.stop()

images_set = {it["name"] for it in image_items}
index_set = set(idx_names)

# （安全のため）極小ファイル削除は明示的に実行
with st.expander("管理者向け：images/ のメンテナンス", expanded=False):
    do_delete_small = st.checkbox("極小ファイル（壊れ画像の可能性）を images/ から削除する", value=False)
    if do_delete_small:
        small_items = [it for it in image_items if int(it.get("size", 0)) < SMALL_SIZE_TH]
        if small_items:
            st.warning(f"images/ に極小ファイルが {len(small_items)} 件あります。削除を試みます。")
            deleted = 0
            failed = 0

            for it in small_items:
                meta = gh_get_content_json(f"{IMAGE_FOLDER}/{it['name']}")
                if not meta or "sha" not in meta:
                    failed += 1
                    continue

                ok = gh_delete_file(
                    f"{IMAGE_FOLDER}/{it['name']}",
                    sha=meta["sha"],
                    message=f"Delete too small image: {it['name']}",
                )
                if ok:
                    deleted += 1
                else:
                    failed += 1

            st.info(f"極小ファイル削除：成功 {deleted} / 失敗 {failed}")

            image_items = list_github_images()
            images_set = {it["name"] for it in image_items}

# index と images/ のズレを補正（不足分だけ同期）
missing = sorted(images_set - index_set)
extra = sorted(index_set - images_set)

if extra:
    st.warning(f"インデックスにあるが images/ に無い項目が {len(extra)} 件あります（例: {extra[:3]}）。")

if missing:
    st.info(f"images/ にあるのに index.npz に無い画像が {len(missing)} 枚あります。不足分だけ同期します。")
    with st.spinner("インデックス不足分を同期中..."):
        idx_names, idx_sha_list, idx_embeddings, idx_comments, idx_file_sha, added, ok = sync_index_missing_images(
            image_items, idx_names, idx_sha_list, idx_embeddings, idx_file_sha, idx_comments
        )
    if ok:
        st.success(f"index.npz に不足分 {added} 枚を追加しました。")
    else:
        st.warning("不足分の同期で一部失敗しました（壊れ画像・取得失敗などの可能性）。")
else:
    st.write("✅ images/ と index.npz は同期済みです（過不足なし）。")

st.write(f"インデックス登録数: {len(idx_names)} 枚")

uploaded_file = st.file_uploader(
    "アップロードファイル（空の写真：JPEG/PNG/HEIC）",
    type=["jpg", "jpeg", "png", "heic", "HEIC"],
)

if not uploaded_file:
    st.caption("特徴量は features/index.npz に集約し、GitHub上のデータだけで類似検索します。")
    st.stop()

image_bytes = uploaded_file.read()
if len(image_bytes) < SMALL_SIZE_TH:
    st.error(f"アップロードされたデータが小さすぎます（{len(image_bytes)} bytes）。アップロード失敗の可能性があります。")
    st.stop()

# ① プレビュー
try:
    preview = load_image_from_bytes(image_bytes)
    st.image(preview, caption="アップロード画像", use_container_width=True)
except Exception as e:
    st.error(f"画像の読み込みに失敗しました: {e}")
    st.stop()

# ② クエリ特徴量
with st.spinner("アップロード画像の特徴量を抽出中..."):
    q_vec = get_image_feature(image_bytes)
q_sha = sha256_hex(image_bytes)

# ③ 類似検索（既存ロジック）
with st.spinner("類似画像を検索中..."):
    results, sims = find_top_similar(q_vec, idx_names, idx_embeddings, TOP_K)
name_to_comment = {n: c for n, c in zip(idx_names, idx_comments)}

st.subheader("🌈 類似している空（上位3枚）")
st.caption("アップロード画像の特徴量と、GitHubに蓄積された特徴量を比較して、見た目が近い空の写真を表示します。")
for name, score in results:
    url = image_download_url_by_name(name, image_items)
    if not url:
        st.warning(f"{name} のURLが見つかりませんでした。")
        continue
    try:
        b = gh_download_bytes_cached(url)
        img = load_image_from_bytes(b)
        comment = name_to_comment.get(name, "")
        cap = f"{name} / 類似度: {score*100:.1f}%"
        if comment:
            cap = cap + f"\n{comment}"
        st.image(img, caption=cap, use_container_width=True)
    except Exception as e:
        st.warning(f"{name} の表示に失敗しました: {e}")
        st.markdown(f"[画像リンク]({url})")

best_score = float(np.max(sims)) if len(sims) else 0.0

# ④ GitHub保存（既存ロジックを基本維持）
st.markdown("---")
st.subheader("📌 この画像を共有（GitHubへ保存）")
st.caption("ここで保存すると、画像と特徴量がGitHubに追加され、次回以降の類似検索で使われます。重複画像は自動で弾きます。")

# --- コメント（画像と紐づけて保存）---
comment_name = st.text_input("本名またはニックネーム（任意）", key="comment_name")
comment_place = st.text_input("撮影場所（任意）", key="comment_place")
comment_note = st.text_area("そのほか残したいこと（任意）", key="comment_note", height=80)

comment_lines = []
if comment_name.strip():
    comment_lines.append(f"名前: {comment_name.strip()}")
if comment_place.strip():
    comment_lines.append(f"場所: {comment_place.strip()}")
if comment_note.strip():
    comment_lines.append(f"メモ: {comment_note.strip()}")
comment_text = "\n".join(comment_lines)

st.write(f"重複判定（最大類似度）: {best_score*100:.1f}%（閾値 {DUP_TH*100:.0f}%）")

save_ok = True
if best_score >= DUP_TH:
    st.warning("⚠️ 既存画像と非常に似ているため、重複の可能性が高いです。保存は行いません。")
    save_ok = False

if q_sha in idx_sha_list:
    st.warning("⚠️ 同一の画像（完全一致）がすでに登録されています。保存は行いません（検索は続行できます）。")
    save_ok = False

if save_ok:
    ext = uploaded_file.name.split(".")[-1].lower()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    img_path = f"{IMAGE_FOLDER}/{filename}"

    if st.button("GitHubに保存する"):
        with st.spinner("GitHubに画像を保存中..."):
            ok_img, uploaded_sha = gh_put_file(
                img_path,
                image_bytes,
                message=f"Add new sky image: {filename}",
                sha=None,
            )

        if not ok_img:
            st.error("画像のアップロードに失敗しました。トークン権限・Secrets設定・リポジトリ名を確認してください。")
        else:
            with st.spinner("index.npz を更新中（検索対象に追加）..."):
                ok_idx = append_to_index_and_upload(filename, q_sha, q_vec, new_comment=comment_text)

            if ok_idx:
                st.success("✅ 画像の保存と index.npz の更新が完了しました（検索対象に追加済み）。")
                raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{IMAGE_FOLDER}/{filename}"
                st.markdown(f"[📸 保存された画像を見る]({raw_url})")
                st.info("検索対象の更新は、ページ再読み込み後に反映される場合があります（キャッシュの影響）。")
            else:
                st.warning("⚠️ index.npz の更新に失敗しました。整合性維持のため、画像アップロードを取り消します。")

                rolled_back = False
                if uploaded_sha:
                    with st.spinner("アップロード済み画像を削除して巻き戻し中..."):
                        rolled_back = gh_delete_file(
                            img_path,
                            sha=uploaded_sha,
                            message=f"Rollback image upload because index update failed: {filename}",
                        )
                else:
                    st.error("画像のshaが取得できなかったため、自動巻き戻しができません。後で同期/再生成を行ってください。")

                if rolled_back:
                    st.info("✅ 画像の巻き戻し（削除）が完了しました。")
                else:
                    st.error("❌ 画像の巻き戻しに失敗しました。images/ と index.npz がズレた可能性があります。後で同期/再生成を行ってください。")

# ============================================================
# ⑤ 単語生成（任意）→ ⑥ スコアリング → ⑧ ボタン押下で描画
# ============================================================
st.markdown("---")
st.subheader("🪄 単語の用意（Gemini または手入力）")
st.caption("Geminiで自動生成するか、手入力で単語を用意します。生成結果は画像のSHAごとにキャッシュされ、同じ画像では繰り返し呼び出しません。")

cache = st.session_state["gemini_cache"]
cached_words = cache.get(q_sha, {}).get("words", [])
cached_defs = cache.get(q_sha, {}).get("defs", {})
cached_raw = cache.get(q_sha, {}).get("raw", "")

colA, colB = st.columns([1, 1])
with colA:
    st.warning("Geminiボタンは連打しないでください。混雑（503）やクォータ上限（429）で失敗しやすくなります。")
    if st.button("Geminiで単語生成（ボタン押下時のみ）"):
        with st.spinner("Geminiで出力生成中..."):
            words, defs, raw = gemini_generate_words(image_bytes)
        if words:
            cache[q_sha] = {"words": words, "defs": defs, "raw": raw}
            # 手入力欄には「単語だけ」を入れる（解説文は混ぜない）
            st.session_state["manual_words_text"] = "、".join(words)
            st.success(f"単語を {len(words)} 個生成しました。")
            st.rerun()
        else:
            st.error(raw)

with colB:
    st.caption("Geminiを使わない場合は、下で手入力してください。")

default_manual = "、".join(cached_words) if cached_words else ""
if "manual_words_text" not in st.session_state:
    st.session_state["manual_words_text"] = default_manual
# Gemini生成後にここへ書き戻すことで、1回の押下でテキストボックスにも反映されます
manual_text = st.text_input(
    "手入力（区切り：読点・カンマ・改行）",
    key="manual_words_text",
    placeholder="例：快晴、うす雲、きらきら、ふわふわ",
)

raw_words: List[str]
if manual_text.strip():
    raw_words = [w.strip() for w in re.split(r"[、,\n]\s*", manual_text) if w.strip()]
else:
    raw_words = cached_words

max_words = st.slider("最大単語数", min_value=1, max_value=12, value=8)
raw_words = raw_words[:max_words]

if cached_defs:
    with st.expander("Geminiによる空の解説", expanded=False):
        for w in cached_words:
            d = str(cached_defs.get(w, "")).strip()
            if d:
                st.write(f"- **{w}**：{d}")
            else:
                st.write(f"- **{w}**")

if cached_raw:
    with st.expander("Geminiの生出力（キャッシュ）", expanded=False):
        st.code(cached_raw)

# ⑥ スコアリング（CLIP）
word_scores = score_words_by_clip(q_vec, raw_words)

if word_scores:
    st.markdown("**CLIP 一致度（上位）**")
    for w, s in word_scores[: min(len(word_scores), 10)]:
        st.write(f"- {w}: {s*100:.1f}%")
else:
    st.info("単語が未設定です。Gemini生成または手入力をしてください。")

# ⑧ 描画（ボタン押下）
st.markdown("---")
st.subheader("🖼️ 文字入り画像（プレビュー）")
st.caption("用意した単語を、画像の上に自動配置して描画します。")

# 描画パラメータ（UIからは変更しない）
FONT_PATH = "fonts/NotoSansJP-Bold.ttf"
PADDING_RATIO = 0.05
MIN_FONT = 0
MAX_FONT = 0
SIZE_GAMMA = 2.2
SIZE_CONTRAST = 5.0
SCATTER_STRENGTH = 0.55

draw_ok = bool(word_scores)

if st.button("文字を画像に描画する", disabled=not draw_ok):
    try:
        with st.spinner("描画中..."):
            key = overlay_cache_key(q_sha, [w for w, _ in word_scores[:max_words]])
            key = f"{key}:min{MIN_FONT}:max{MAX_FONT}:g{SIZE_GAMMA:.2f}:ct{SIZE_CONTRAST:.2f}:sc{SCATTER_STRENGTH:.2f}"
            overlay_cache = st.session_state["overlay_cache"]
            if key not in overlay_cache:
                overlay_cache[key] = overlay_words_variable_size(
                    image_bytes=image_bytes,
                    word_scores=word_scores[:max_words],
                    font_path=FONT_PATH,
                    padding_ratio=PADDING_RATIO,
                    min_font=MIN_FONT,
                    max_font=MAX_FONT,
                    size_gamma=SIZE_GAMMA,
                    size_contrast=SIZE_CONTRAST,
                    scatter_strength=SCATTER_STRENGTH,
                    seed=int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16),
                )
            st.session_state["overlay_last_key"] = key
        st.success("描画しました（session_state に保持）。")
    except Exception as e:
        st.error(f"描画に失敗しました: {e}")

last_key = st.session_state.get("overlay_last_key")
if last_key and last_key in st.session_state["overlay_cache"]:
    overlay_img: Image.Image = st.session_state["overlay_cache"][last_key]
    st.image(overlay_img, use_container_width=True)

    buf = io.BytesIO()
    overlay_img.save(buf, format="JPEG", quality=95)
    img_data = buf.getvalue()

    st.download_button(
        label="📥 画像をダウンロード",
        data=img_data,
        file_name="soramoyou_share.jpg",
        mime="image/jpeg",
    )

st.markdown("---")
st.caption("特徴量は features/index.npz に集約し、GitHub上のデータだけで類似検索します。")