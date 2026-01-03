# streamlit run visualize_wrong_answer.py

import streamlit as st
import pandas as pd
import json
import os
import mimetypes
import time
import tempfile

# Google Gen AI SDK
# pip install google-genai
from google import genai
from google.genai import types

# =========================
# Page
# =========================
st.set_page_config(layout="wide", page_title="Wrong Answer Inspector (Video → Gemini)")

# =========================
# Data Loading
# =========================
@st.cache_data
def load_data():
    csv_path = os.path.join("answers", "sutd_videochat_wrong_ans_yolo_no_prompting.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df_wrong = pd.read_csv(csv_path)

    jsonl_path = os.path.join("answers", "sutd_test_gt.jsonl")
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()

    data = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            headers = json.loads(f.readline())
            for line in f:
                if line.strip():
                    values = json.loads(line)
                    # Some jsonl are list-rows with a header row at top
                    if isinstance(values, list) and len(values) == len(headers):
                        data.append(dict(zip(headers, values)))
                    # Some jsonl are dict-rows already
                    elif isinstance(values, dict):
                        data.append(values)
    except Exception:
        # keep app alive even if GT parsing fails
        pass

    df_gt = pd.DataFrame(data)

    df_wrong["id"] = pd.to_numeric(df_wrong["id"], errors="coerce").fillna(0).astype(int)
    if "record_id" in df_gt.columns:
        df_gt["record_id"] = pd.to_numeric(df_gt["record_id"], errors="coerce").fillna(0).astype(int)

    if "record_id" in df_gt.columns:
        return pd.merge(df_wrong, df_gt, left_on="id", right_on="record_id", how="left")

    return df_wrong


df = load_data()
if df.empty:
    st.error("Could not load data. Please check file paths.")
    st.stop()

# IMPORTANT: make ids pure python ints (avoids widget/state weirdness)
ids = [int(x) for x in df["id"].tolist()]

# =========================
# Prompt
# =========================
def build_short_prompt(row: pd.Series) -> str:
    try:
        gt_idx = int(row.get("gt_answer", -1))
    except Exception:
        gt_idx = -1

    try:
        pred_idx = int(row.get("pred_answer", -1))
    except Exception:
        pred_idx = -1

    q = row.get("q_body", "N/A")

    options = []
    for i in range(4):
        opt = row.get(f"option{i}", "N/A")
        if pd.isna(opt):
            opt = "N/A"
        options.append(f"{i}. {opt}")

    pred_text = row.get(f"option{pred_idx}", "N/A") if 0 <= pred_idx < 4 else "N/A"
    gt_text = row.get(f"option{gt_idx}", "N/A") if 0 <= gt_idx < 4 else "N/A"

    return f"""You are given a VIDEO of the scene/question. Use the video to reason.
Write EXACTLY 1 complete short sentence (no bullets, no fragments).
Explain the most likely reason the model picked the predicted option instead of the ground truth.

Question: {q}

Options:
{chr(10).join(options)}

Predicted: ({pred_idx}) {pred_text}
Ground truth: ({gt_idx}) {gt_text}
"""


# =========================
# Helpers: bytes + debug
# =========================
@st.cache_data(show_spinner=False)
def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _resp_to_dict(resp):
    """Best-effort conversion to a JSON-serializable dict for debug display."""
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
    except Exception:
        pass
    try:
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
    except Exception:
        pass
    try:
        if hasattr(resp, "model_dump_json"):
            return json.loads(resp.model_dump_json())
    except Exception:
        pass
    return {"repr": repr(resp)}

# =========================
# Gemini call: Upload Video -> Generate
# =========================
def call_gemini_with_inline_video(
    api_key: str,
    model: str,
    prompt_text: str,
    video_bytes: bytes,
    video_mime: str,
    temperature: float = 0.2,
) -> tuple[str, str, dict]:
    """
    Uploads video to Gemini File API (required for videos), waits for processing, 
    then generates content.
    Returns (text, finish_reason-ish, raw_payload_dict)
    """
    client = genai.Client(api_key=api_key)

    # 1. Write bytes to a temp file so we can upload it via path
    # (The SDK upload method works best with file paths)
    ext = ".mp4"
    if "webm" in video_mime: ext = ".webm"
    elif "avi" in video_mime: ext = ".avi"
    elif "mov" in video_mime: ext = ".mov"
    
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        tf.write(video_bytes)
        tf_path = tf.name

    try:
        print(f"Uploading {len(video_bytes)} bytes to Gemini Files API...")
        # Upload the file
        video_file = client.files.upload(file=tf_path)
        
        # 2. Wait for the video to process (CRITICAL STEP)
        # Videos enter 'PROCESSING' state. We must wait for 'ACTIVE'.
        print(f"File uploaded: {video_file.name}. Waiting for processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        print("Video is active. Generating content...")

        # 3. Generate Content
        # We pass the 'video_file' object directly in the contents list.
        # The SDK handles the reference.
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part(text=prompt_text),
                video_file
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )

        print("Gemini response received.")
        
        # --- Parsing Logic (Same as before) ---
        raw = _resp_to_dict(resp)
        text = (getattr(resp, "text", None) or "").strip()

        finish = ""
        try:
            if getattr(resp, "candidates", None) and len(resp.candidates) > 0:
                finish = getattr(resp.candidates[0], "finish_reason", "") or ""
        except Exception:
            finish = ""

        # Fallback if text empty
        if not text:
            try:
                cand0 = resp.candidates[0] if getattr(resp, "candidates", None) else None
                parts = cand0.content.parts if (cand0 and getattr(cand0, "content", None)) else []
                texts = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
                text = "\n".join(texts).strip()
            except Exception:
                text = ""

        return text or "(No text returned)", str(finish or ""), raw

    finally:
        # Cleanup: remove the local temp file
        if os.path.exists(tf_path):
            os.remove(tf_path)

# =========================
# Session state (robust navigation; no reset on Evaluate)
# =========================
if "selected_record" not in st.session_state:
    st.session_state.selected_record = ids[0]

if "current_index" not in st.session_state:
    st.session_state.current_index = ids.index(int(st.session_state.selected_record))

# Separate widget key so the widget doesn't “own” your source-of-truth
if "selected_record_widget" not in st.session_state:
    st.session_state.selected_record_widget = int(st.session_state.selected_record)

# Cache Gemini results per ID
# {id: {"text": "...", "finish": "...", "model": "...", "raw": {...}}}
if "gemini_results" not in st.session_state:
    st.session_state.gemini_results = {}

def sync_from_widget():
    val = int(st.session_state.selected_record_widget)
    st.session_state.selected_record = val
    st.session_state.current_index = ids.index(val)

def go_prev():
    idx = (st.session_state.current_index - 1) % len(ids)
    st.session_state.current_index = idx
    st.session_state.selected_record = ids[idx]
    st.session_state.selected_record_widget = ids[idx]

def go_next():
    idx = (st.session_state.current_index + 1) % len(ids)
    st.session_state.current_index = idx
    st.session_state.selected_record = ids[idx]
    st.session_state.selected_record_widget = ids[idx]


# =========================
# Current row + video path (single source for UI + Evaluate)
# =========================
current_id = int(st.session_state.selected_record)
row = df.loc[df["id"] == current_id].iloc[0]

fname = row["filename"] if pd.notna(row.get("filename")) else row.get("vid_filename", "")
video_path = os.path.join("sutd-traffic-video-qa", "videos_obj_tracking", "videos_obj_tracking", str(fname))
video_exists = bool(fname) and os.path.exists(video_path)

# Guess MIME from extension; default mp4
video_mime = mimetypes.guess_type(video_path)[0] if video_exists else None
if not video_mime:
    video_mime = "video/mp4"

st.sidebar.title("Gemini evaluation")

api_key_value = st.secrets["api_key"] if "api_key" in st.secrets else ""
api_key = st.sidebar.text_input("API key", value=api_key_value, type="password", placeholder="Paste your key")

model_name = st.sidebar.selectbox(
    "Model",
    options=[
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ],
    index=0,
)

temperature = 0.2
with st.sidebar.expander("Advanced", expanded=False):
    temperature = st.slider("Temperature", 0.0, 1.0, temperature, 0.05)

st.sidebar.caption(f"Current ID: {current_id}")
st.sidebar.caption(f"Video: {fname if fname else '(none)'}")

eval_disabled = (not api_key.strip()) or (not video_exists)
eval_btn = st.sidebar.button("🤖 Evaluate (send video)", use_container_width=True, disabled=eval_disabled)

cached = current_id in st.session_state.gemini_results
clear_btn = st.sidebar.button("🧹 Clear (this ID)", use_container_width=True, disabled=not cached)

if clear_btn:
    st.session_state.gemini_results.pop(current_id, None)

if eval_btn:
    prompt = build_short_prompt(row)

    try:
        with st.sidebar.status("Evaluating…", expanded=True):
            st.write("Video path:", video_path)
            st.write("MIME:", video_mime)

            # Read bytes (cached by path)
            vb = read_bytes(video_path)
            st.write("Video size (bytes):", len(vb))

            st.write("Prompt preview:")
            st.code(prompt[:1200] + ("..." if len(prompt) > 1200 else ""))

            text, finish, raw = call_gemini_with_inline_video(
                api_key=api_key.strip(),
                model=model_name,
                prompt_text=prompt,
                video_bytes=vb,
                video_mime=video_mime,
                temperature=temperature,
            )

        st.session_state.gemini_results[current_id] = {
            "text": text,
            "finish": finish,
            "model": model_name,
            "raw": raw,
        }

    except Exception as e:
        st.sidebar.error(f"Gemini call failed: {e}")

st.sidebar.divider()
st.sidebar.subheader("Result")

res = st.session_state.gemini_results.get(current_id)
if res:
    finish = (res.get("finish") or "").strip()
    if finish and finish.upper() != "STOP":
        st.sidebar.warning(f"Stopped early: {finish}")
    st.sidebar.caption(f"Model: {res.get('model', '')}")
    st.sidebar.write(res.get("text", ""))

    with st.sidebar.expander("Debug: raw Gemini response", expanded=False):
        st.json(res.get("raw", {}))
else:
    if not video_exists:
        st.sidebar.warning("Video file not found for this record, so Evaluate is disabled.")
    else:
        st.sidebar.caption("Click **Evaluate (send video)** to get a sentence explanation.")


# =========================
# Main UI
# =========================
st.title("Wrong Answer Inspector")

top_left, top_mid, top_right = st.columns([1, 3, 1], vertical_alignment="center")

with top_left:
    c1, c2 = st.columns(2)
    with c1:
        st.button("◀ Prev", on_click=go_prev, use_container_width=True)
    with c2:
        st.button("Next ▶", on_click=go_next, use_container_width=True)

with top_mid:
    st.selectbox(
        "Record",
        options=ids,
        # index=ids.index(int(st.session_state.selected_record)),
        key="selected_record_widget",
        on_change=sync_from_widget,
        label_visibility="collapsed",
    )

with top_right:
    st.caption(f"ID: **{current_id}**")

st.divider()

col_video, col_details = st.columns([1, 1], gap="large")

with col_video:
    st.subheader("Video")
    if video_exists:
        st.video(video_path)
        st.caption(f"`{fname}`")
    else:
        st.warning(f"Video not found: `{fname}`")

with col_details:
    st.subheader("Question")
    st.write(row.get("q_body", "N/A"))

    try:
        gt_idx = int(row.get("gt_answer", -1))
    except Exception:
        gt_idx = -1
    try:
        pred_idx = int(row.get("pred_answer", -1))
    except Exception:
        pred_idx = -1

    st.subheader("Answers")
    for i in range(4):
        option_text = row.get(f"option{i}", "N/A")
        if pd.isna(option_text):
            option_text = "N/A"

        icon = ""
        if i == gt_idx and i == pred_idx:
            icon = "✅ GT+Pred"
        elif i == gt_idx:
            icon = "✅ GT"
        elif i == pred_idx:
            icon = "❌ Pred"

        st.markdown(
            f"""
            <div style="padding: 10px 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid #e6e6e6;">
              <div style="font-weight:600; font-size:0.92em; margin-bottom:4px;">
                Option {i} <span style="opacity:0.75;">{icon}</span>
              </div>
              <div style="opacity:0.95;">{option_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
