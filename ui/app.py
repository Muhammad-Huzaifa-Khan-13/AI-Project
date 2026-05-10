from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Dict

# `streamlit run ui/app.py` loads this file as __main__, so relative imports fail.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import streamlit as st

from src.model_b_distractors import generate_row_distractors
from src.model_b_hints import generate_row_hints
from src.model_a_inference import predict_single_row_label

from ui.ui_utils import (
    LABELS,
    confusion_matrix_figure,
    get_row_by_selector,
    load_split_cached,
    model_comparison_bar_figure,
    read_json,
)

APP_TITLE = "Intelligent Reading Comprehension & Quiz Generator"
APP_SUBTITLE = "AI-Powered Reading Comprehension System"
FOOTER_LINES = (
    "Made by Abdul Moiz (23I-0722) & Huzaifa Khan (23I-0635)",
    "FAST NUCES — AI Lab Project",
)

NAV_ITEMS = [
    ("🏠 Home", "Home"),
    ("📝 Quiz Lab", "Quiz"),
    ("📊 Analytics", "Analytics"),
]

PREMIUM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  :root {
    --bg-deep: #050508;
    --bg-mid: #0a0c12;
    --glass: rgba(22, 27, 38, 0.55);
    --glass-border: rgba(139, 168, 255, 0.12);
    --text: #e8edf5;
    --text-muted: rgba(200, 210, 230, 0.72);
    --accent: #7c9cff;
    --accent2: #c084fc;
    --glow: rgba(124, 156, 255, 0.35);
    --success: #4ade80;
    --radius: 18px;
    --font: 'Outfit', 'DM Sans', system-ui, sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--font) !important;
  }

  .stApp {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(124, 156, 255, 0.18), transparent 50%),
                radial-gradient(ellipse 80% 50% at 100% 50%, rgba(192, 132, 252, 0.08), transparent 45%),
                radial-gradient(ellipse 60% 40% at 0% 80%, rgba(56, 189, 248, 0.06), transparent 40%),
                linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 45%, #060810 100%) !important;
    color: var(--text);
  }

  #MainMenu { visibility: hidden; height: 0; }
  footer { visibility: hidden; height: 0; }
  header[data-testid="stHeader"] { background: transparent !important; }

  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1280px !important;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(165deg, rgba(12, 14, 22, 0.92) 0%, rgba(8, 10, 18, 0.96) 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
    box-shadow: 8px 0 40px rgba(0, 0, 0, 0.35);
  }

  [data-testid="stSidebar"] .stMarkdown { color: var(--text); }

  [data-testid="stVerticalBlock"] > div:has(> div > div > label[data-testid="stWidgetLabel"]) label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
  }

  div[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-weight: 500 !important;
  }

  .stSlider label { color: var(--text-muted) !important; }

  div[data-baseweb="select"] > div {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
  }

  div[data-baseweb="input"] > div {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
  }

  .stTextInput input, .stNumberInput input {
    color: var(--text) !important;
    background: rgba(15, 18, 28, 0.6) !important;
  }

  /* Buttons — premium hover */
  .stButton > button {
    font-family: var(--font) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    border-radius: 14px !important;
    border: 1px solid rgba(124, 156, 255, 0.35) !important;
    background: linear-gradient(135deg, rgba(124, 156, 255, 0.25) 0%, rgba(192, 132, 252, 0.2) 100%) !important;
    color: var(--text) !important;
    padding: 0.65rem 1.25rem !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }

  .stButton > button:hover {
    transform: translateY(-3px);
    border-color: rgba(192, 132, 252, 0.55) !important;
    box-shadow: 0 12px 36px var(--glow), 0 0 0 1px rgba(255,255,255,0.06) inset;
  }

  .stButton > button:active {
    transform: translateY(-1px);
  }

  .stButton > button[kind="primary"],
  .stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.45);
  }

  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 14px 44px rgba(139, 92, 246, 0.55);
  }

  /* Alerts / info boxes */
  div[data-testid="stAlert"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(12, 14, 22, 0.5);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid var(--glass-border);
  }

  .stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-muted);
    font-weight: 500;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(168, 85, 247, 0.25)) !important;
    color: var(--text) !important;
  }

  /* Custom HTML blocks */
  .hero-wrap {
    position: relative;
    border-radius: 24px;
    padding: 2.5rem 2.75rem;
    margin-bottom: 2rem;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255,255,255,0.04) inset;
  }

  .hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(125deg, rgba(99, 102, 241, 0.45) 0%, rgba(168, 85, 247, 0.28) 35%, rgba(56, 189, 248, 0.15) 100%);
    opacity: 0.95;
    z-index: 0;
  }

  .hero-wrap::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 120%, rgba(255,255,255,0.15), transparent 55%);
    z-index: 0;
    pointer-events: none;
  }

  .hero-inner { position: relative; z-index: 1; }

  .hero-title {
    font-family: var(--font);
    font-size: clamp(1.65rem, 3.5vw, 2.35rem);
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.03em;
    line-height: 1.2;
    text-shadow: 0 2px 24px rgba(0,0,0,0.35);
    margin: 0 0 0.5rem 0;
  }

  .hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.88);
    font-weight: 500;
    margin: 0 0 1rem 0;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    background: rgba(0,0,0,0.22);
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.92);
    font-size: 0.8rem;
    font-weight: 600;
  }

  .glass-card {
    background: var(--glass);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.35rem 1.5rem;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.28), 0 0 0 1px rgba(255,255,255,0.03) inset;
  }

  .glass-card h3, .glass-card .card-title {
    font-size: 0.92rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    margin: 0 0 0.85rem 0;
  }

  .article-scroll {
    max-height: 340px;
    overflow-y: auto;
    padding-right: 0.75rem;
    color: var(--text);
    line-height: 1.65;
    font-size: 0.98rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(124, 156, 255, 0.45) transparent;
  }

  .article-scroll::-webkit-scrollbar { width: 6px; }
  .article-scroll::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    border-radius: 6px;
  }

  .question-block {
    margin-top: 1.1rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--glass-border);
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text);
    line-height: 1.55;
  }

  .mcq-grid { display: flex; flex-direction: column; gap: 0.75rem; }

  .mcq-option {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1rem 1.15rem;
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(22, 27, 38, 0.75), rgba(15, 18, 28, 0.65));
    border: 1px solid var(--glass-border);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }

  .mcq-option:hover {
    transform: translateX(4px);
    border-color: rgba(124, 156, 255, 0.35);
    box-shadow: 0 8px 28px rgba(124, 156, 255, 0.12);
  }

  .mcq-letter {
    flex-shrink: 0;
    width: 2rem;
    height: 2rem;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
  }

  .mcq-text { color: var(--text); line-height: 1.5; font-size: 0.95rem; }

  .correct-pill {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(34, 197, 94, 0.08));
    border: 1px solid rgba(74, 222, 128, 0.35);
    color: #86efac;
    font-weight: 600;
    font-size: 0.9rem;
  }

  .section-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }

  .page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
  }

  .page-caption { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem; }

  .footer-brand {
    margin-top: 3rem;
    padding: 1.25rem 1.5rem;
    border-radius: var(--radius);
    border: 1px solid var(--glass-border);
    background: rgba(12, 14, 22, 0.55);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.65;
  }

  .footer-brand strong { color: var(--text); font-weight: 600; }

  .hint-tier {
    margin-top: 0.75rem;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    background: rgba(15, 18, 28, 0.5);
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.55;
  }

  .hint-tier-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.35rem;
  }

  .sidebar-brand {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.3;
    margin-bottom: 0.25rem;
  }

  .sidebar-sub { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 1rem; }
</style>
"""


def inject_premium_theme() -> None:
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


def set_theme() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_premium_theme()


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-brand">
          <strong>{html.escape(FOOTER_LINES[0])}</strong><br/>
          {html.escape(FOOTER_LINES[1])}
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_model_a_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def predict_model_a(model_artifact, row) -> str:
  # New artifacts are dictionaries with option-wise scoring components.
  if isinstance(model_artifact, dict) and model_artifact.get("kind") == "optionwise_binary":
    return predict_single_row_label(model_artifact, row)

  # Legacy compatibility: old direct-label pipeline on verifier_input.
  verifier_input = str(row.get("verifier_input", ""))
  pred = model_artifact.predict([verifier_input])[0]
  return str(pred).strip().upper()


def home_page() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-inner">
            <h1 class="hero-title">📘 {html.escape(APP_TITLE)}</h1>
            <p class="hero-sub">✨ {html.escape(APP_SUBTITLE)}</p>
            <span class="hero-badge">🧠 Model A Verifier · Model B Distractors &amp; Hints · RACE Dataset</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        st.markdown(
            """
            <div class="glass-card">
              <div class="card-title">🎯 Capabilities</div>
              <p style="color: rgba(200,210,230,0.85); line-height: 1.65; margin: 0; font-size: 0.95rem;">
                Browse processed RACE samples · Run the verifier · Generate intelligent distractors ·
                Progressive hints (easy → medium → hard) · Full analytics dashboard
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
              <div class="card-title">✅ Stack</div>
              <p style="color: rgba(200,210,230,0.85); line-height: 1.65; margin: 0; font-size: 0.95rem;">
                Preprocessing complete · Model A (TF‑IDF + classical ML) · Model B (TF‑IDF ranking + extractive hints) ·
                End-to-end <span style="font-family:monospace;font-size:0.88rem;">demo_pipeline.py</span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="glass-card">
              <div class="card-title">🚀 Run</div>
              <p style="color: rgba(200,210,230,0.85); line-height: 1.65; margin: 0; font-size: 0.95rem;">
                <code style="background:rgba(0,0,0,0.35);padding:0.2rem 0.45rem;border-radius:6px;">venv\\Scripts\\python -m streamlit run ui\\app.py</code>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _mcq_options_html(options: Dict[str, str]) -> str:
    parts = []
    for k in LABELS:
        letter = html.escape(k)
        text = html.escape(options.get(k, ""))
        parts.append(
            f'<div class="mcq-option"><span class="mcq-letter">{letter}</span><span class="mcq-text">{text}</span></div>'
        )
    return '<div class="mcq-grid">' + "".join(parts) + "</div>"


def quiz_page() -> None:
    st.markdown('<p class="section-label">Interactive demo</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="page-title">📝 Quiz Lab</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-caption">Load a sample from the sidebar, then run Model A, generate distractors, and reveal hints step by step.</p>',
        unsafe_allow_html=True,
    )

    split = st.session_state.get("split", "val")
    df = load_split_cached(split)

    row = st.session_state.get("current_row")
    if row is None:
        row = df.sample(n=1, random_state=int(st.session_state.get("seed", 42))).iloc[0]
        st.session_state["current_row"] = row

    # Navigation controls: Previous / Next sample (By Index)
    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 4])
    with nav_c1:
      if st.button("◀ Previous", use_container_width=True):
        # ensure row_idx present
        idx = int(st.session_state.get("row_idx", 0))
        idx = max(0, idx - 1)
        st.session_state["row_idx"] = int(idx)
        st.session_state["current_row"] = get_row_by_selector(df, mode="By Index", row_idx=idx, example_id="", seed=int(st.session_state.get("seed", 42)))
        st.session_state.pop("model_a_pred", None)
        st.session_state.pop("distractors", None)
        st.session_state.pop("hints", None)
        st.session_state["hint_level"] = 0
    with nav_c2:
      if st.button("Next ▶", use_container_width=True):
        idx = int(st.session_state.get("row_idx", 0))
        idx = min(len(df) - 1, idx + 1)
        st.session_state["row_idx"] = int(idx)
        st.session_state["current_row"] = get_row_by_selector(df, mode="By Index", row_idx=idx, example_id="", seed=int(st.session_state.get("seed", 42)))
        st.session_state.pop("model_a_pred", None)
        st.session_state.pop("distractors", None)
        st.session_state.pop("hints", None)
        st.session_state["hint_level"] = 0
    with nav_c3:
      st.markdown(f"**Row:** {int(st.session_state.get('row_idx', 0))} / {len(df)-1}")

    article = str(row.get("article", ""))
    question = str(row.get("question", ""))
    options = {k: str(row.get(k, "")) for k in LABELS}
    correct_label = str(row.get("answer", "")).strip().upper()
    correct_text = options.get(correct_label, "")
    sample_id = html.escape(str(row.get("id", "—")))

    top = st.columns([1.55, 1])
    with top[0]:
        st.markdown('<div class="section-label">Reading passage</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="card-title">📄 Article <span style="opacity:0.7;font-weight:500;letter-spacing:0;">· id {sample_id}</span></div>
              <div class="article-scroll">{html.escape(article)}</div>
              <div class="question-block">❓ {html.escape(question)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top[1]:
        st.markdown('<div class="section-label">Multiple choice</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="card-title">✅ Options</div>
              {_mcq_options_html(options)}
              <div class="correct-pill">🎯 Correct: <strong>{html.escape(correct_label)}</strong>) {html.escape(correct_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns([1, 1, 1])

    with a1:
        st.markdown('<div class="section-label">Verifier</div>', unsafe_allow_html=True)
        st.caption("Model A — predict A/B/C/D from `verifier_input`.")
        if st.button("▶ Run Model A Prediction", type="primary", use_container_width=True):
            artifact_path = Path(st.session_state.get("model_a_path", "models/model_a/traditional/linear_svm.joblib"))
            model = load_model_a_artifact(artifact_path)
            if model is None:
                st.error(
                    f"Model A artifact not found at `{artifact_path}`. Train first:\n"
                    f"`.\u005cvenv\u005cScripts\u005cpython -m src.model_a_train`"
                )
            else:
                verifier_input = str(row.get("verifier_input", ""))
                if not verifier_input.strip():
                    st.error("This row has empty `verifier_input`. Re-run preprocessing or pick another sample.")
                else:
                  st.session_state["model_a_pred"] = predict_model_a(model, row)

    with a2:
        st.markdown('<div class="section-label">Distractors</div>', unsafe_allow_html=True)
        st.caption("Model B — top-3 diverse distractors (correct excluded).")
        if st.button("✨ Generate Distractors", use_container_width=True):
            st.session_state["distractors"] = generate_row_distractors(
                row=row,
                max_features=int(st.session_state.get("max_features_b", 5000)),
                diversity_threshold=float(st.session_state.get("div_threshold", 0.85)),
                max_sentences=int(st.session_state.get("max_sentences", 20)),
            )

    with a3:
        st.markdown('<div class="section-label">Hints</div>', unsafe_allow_html=True)
        st.caption("Model B — progressive extractive hints.")
        if st.button("💡 Generate Hints", use_container_width=True):
            st.session_state["hints"] = generate_row_hints(
                row=row,
                max_features=int(st.session_state.get("max_features_b", 5000)),
                max_sentences=int(st.session_state.get("max_sentences_hints", 30)),
            )
            st.session_state["hint_level"] = 0

    st.markdown("<br/>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1])

    with r1:
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🤖 Model A &amp; Distractors</div>', unsafe_allow_html=True)

        pred_label = st.session_state.get("model_a_pred")
        if pred_label:
            pred_text = options.get(str(pred_label).strip().upper(), "")
            st.success(f"**Prediction:** `{pred_label}` — {pred_text}")
        else:
            st.info("Run Model A to see a prediction.")

        d = st.session_state.get("distractors")
        if d:
            st.markdown("**Generated distractors**")
            for i, key in enumerate(["distractor_1", "distractor_2", "distractor_3"], start=1):
                st.markdown(f"{i}. {d.get(key, '')}")
        else:
            st.caption("Generate distractors to display them here.")

        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="section-label">Progressive hints</div>', unsafe_allow_html=True)
        h = st.session_state.get("hints")
        if not h:
            st.info("Generate hints to unlock easy → medium → hard reveal.")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔮 Hint reveal</div>', unsafe_allow_html=True)

            level = int(st.session_state.get("hint_level", 0))
            c_e, c_m, c_h = st.columns(3)
            with c_e:
                if st.button("🌤 Easy", use_container_width=True):
                    st.session_state["hint_level"] = max(level, 1)
            with c_m:
                if st.button("⛅ Medium", use_container_width=True, disabled=level < 1):
                    st.session_state["hint_level"] = max(int(st.session_state.get("hint_level", 0)), 2)
            with c_h:
                if st.button("🌩 Hard", use_container_width=True, disabled=level < 2):
                    st.session_state["hint_level"] = max(int(st.session_state.get("hint_level", 0)), 3)

            if st.session_state.get("hint_level", 0) >= 1:
                st.markdown(
                    f'<div class="hint-tier"><div class="hint-tier-label">Easy</div>{html.escape(str(h.get("hint_easy", "")))}</div>',
                    unsafe_allow_html=True,
                )
            if st.session_state.get("hint_level", 0) >= 2:
                st.markdown(
                    f'<div class="hint-tier"><div class="hint-tier-label">Medium</div>{html.escape(str(h.get("hint_medium", "")))}</div>',
                    unsafe_allow_html=True,
                )
            if st.session_state.get("hint_level", 0) >= 3:
                st.markdown(
                    f'<div class="hint-tier"><div class="hint-tier-label">Hard</div>{html.escape(str(h.get("hint_hard", "")))}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


def analytics_page() -> None:
    st.markdown('<p class="section-label">Insights</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="page-title">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-caption">Model A validation metrics, comparison charts, and confusion matrices from saved JSON reports.</p>',
        unsafe_allow_html=True,
    )

    report_dir = Path("models/model_a/reports")
    baseline = read_json(report_dir / "baseline_metrics.json")
    ensemble = read_json(report_dir / "ensemble_metrics.json")
    stacking = read_json(report_dir / "stacking_metrics.json")
    unsupervised = read_json(report_dir / "unsupervised_metrics.json")

    if not baseline and not ensemble and not stacking and not unsupervised:
        st.error("No Model A report JSONs found under `models/model_a/reports/`.")
        return

    comparison_rows = None
    for rep in [stacking, ensemble, unsupervised]:
        if rep and "comparison" in rep and rep["comparison"].get("validation_table_sorted_by_macro_f1"):
            comparison_rows = rep["comparison"]["validation_table_sorted_by_macro_f1"]
            break
    if comparison_rows is None and baseline and "models" in baseline:
        comparison_rows = [
            {
                "model_name": name,
                "validation_accuracy": data["validation_metrics"]["accuracy"],
                "validation_macro_f1": data["validation_metrics"]["macro_f1"],
            }
            for name, data in baseline["models"].items()
        ]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Model comparison (validation)</div>', unsafe_allow_html=True)
    if comparison_rows:
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(model_comparison_bar_figure(comparison_rows, "validation_accuracy"), clear_figure=True)
        with c2:
            st.pyplot(model_comparison_bar_figure(comparison_rows, "validation_macro_f1"), clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧮 Confusion matrices</div>', unsafe_allow_html=True)
    st.caption("Sourced from your evaluation reports.")

    tabs = st.tabs(["Linear SVM", "Logistic Regression", "Ensemble", "Stacking", "KMeans"])

    svm_cm = None
    lr_cm = None
    if baseline and "models" in baseline:
        lr_cm = baseline["models"]["logistic_regression"]["validation_metrics"]["confusion_matrix"]
        svm_cm = baseline["models"]["linear_svm"]["validation_metrics"]["confusion_matrix"]

    with tabs[0]:
        if svm_cm:
            st.pyplot(confusion_matrix_figure(svm_cm, LABELS, "Linear SVM (Validation)"), clear_figure=True)
        else:
            st.info("Linear SVM confusion matrix not found.")
    with tabs[1]:
        if lr_cm:
            st.pyplot(confusion_matrix_figure(lr_cm, LABELS, "Logistic Regression (Validation)"), clear_figure=True)
        else:
            st.info("Logistic Regression confusion matrix not found.")

    with tabs[2]:
        if ensemble and ensemble.get("ensemble", {}).get("validation_metrics", {}).get("confusion_matrix"):
            cm = ensemble["ensemble"]["validation_metrics"]["confusion_matrix"]
            st.pyplot(confusion_matrix_figure(cm, LABELS, "Ensemble (Validation)"), clear_figure=True)
            st.caption(
                f"Disagreement rate (LR vs SVM): {ensemble['ensemble'].get('validation_disagreement_rate_lr_vs_svm', 0.0):.3f}"
            )
        else:
            st.info("Ensemble confusion matrix not found.")

    with tabs[3]:
        if stacking and stacking.get("stacking", {}).get("validation_metrics", {}).get("confusion_matrix"):
            cm = stacking["stacking"]["validation_metrics"]["confusion_matrix"]
            st.pyplot(confusion_matrix_figure(cm, LABELS, "Stacking (Validation)"), clear_figure=True)
        else:
            st.info("Stacking confusion matrix not found.")

    with tabs[4]:
        if unsupervised and unsupervised.get("kmeans", {}).get("validation_metrics", {}).get("confusion_matrix"):
            cm = unsupervised["kmeans"]["validation_metrics"]["confusion_matrix"]
            st.pyplot(confusion_matrix_figure(cm, LABELS, "KMeans (Validation)"), clear_figure=True)
            st.caption(f"Silhouette score: {unsupervised['kmeans'].get('silhouette_score', 0.0):.6f}")
        else:
            st.info("KMeans confusion matrix not found.")

    st.markdown("</div>", unsafe_allow_html=True)


def sidebar() -> str:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">📘 QuizGen AI</div>
        <div class="sidebar-sub">{html.escape(APP_SUBTITLE)}</div>
        """,
        unsafe_allow_html=True,
    )

    labels = [x[0] for x in NAV_ITEMS]
    keys = [x[1] for x in NAV_ITEMS]
    default_key = st.session_state.get("nav_page", "Quiz")
    try:
        default_idx = keys.index(default_key)
    except ValueError:
        default_idx = 1

    choice = st.sidebar.radio("Navigate", labels, index=default_idx)
    page = dict(NAV_ITEMS)[choice]
    st.session_state["nav_page"] = page

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="section-label" style="margin-top:0.5rem;">Dataset</p>', unsafe_allow_html=True)
    split = st.sidebar.selectbox(
        "Split",
        ["train", "val", "test"],
        index=["train", "val", "test"].index(st.session_state.get("split", "val")),
        label_visibility="collapsed",
    )
    st.session_state["split"] = split

    df = load_split_cached(split)
    st.sidebar.caption(f"📑 {len(df):,} rows loaded")

    mode = st.sidebar.selectbox("Selection mode", ["Random", "By Index", "By ID"])
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=int(st.session_state.get("seed", 42)))
    st.session_state["seed"] = int(seed)

    row_idx = st.sidebar.number_input("Row index", min_value=0, max_value=max(0, len(df) - 1), value=int(st.session_state.get("row_idx", 0)))
    example_id = st.sidebar.text_input("Example ID", value=str(st.session_state.get("example_id", "")))
    st.session_state["row_idx"] = int(row_idx)
    st.session_state["example_id"] = str(example_id)

    if st.sidebar.button("⬇ Load sample", use_container_width=True, type="primary"):
        st.session_state["current_row"] = get_row_by_selector(
            df,
            mode=mode,
            row_idx=int(row_idx),
            example_id=str(example_id).strip(),
            seed=int(seed),
        )
        st.session_state.pop("model_a_pred", None)
        st.session_state.pop("distractors", None)
        st.session_state.pop("hints", None)
        st.session_state["hint_level"] = 0

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="section-label">Model controls</p>', unsafe_allow_html=True)
    st.session_state["model_a_path"] = st.sidebar.text_input(
        "Model A `.joblib` path",
        value=str(st.session_state.get("model_a_path", "models/model_a/traditional/linear_svm.joblib")),
        help="Saved pipeline from `python -m src.model_a_train`.",
    )
    st.session_state["max_features_b"] = st.sidebar.slider(
        "Model B max_features", 1000, 20000, int(st.session_state.get("max_features_b", 5000)), step=500
    )
    st.session_state["div_threshold"] = st.sidebar.slider(
        "Distractor diversity", 0.50, 0.99, float(st.session_state.get("div_threshold", 0.85)), step=0.01
    )
    st.session_state["max_sentences"] = st.sidebar.slider(
        "Max sentences (distractors)", 5, 40, int(st.session_state.get("max_sentences", 20))
    )
    st.session_state["max_sentences_hints"] = st.sidebar.slider(
        "Max sentences (hints)", 5, 60, int(st.session_state.get("max_sentences_hints", 30))
    )

    return page


def main() -> None:
    set_theme()
    page = sidebar()

    if page == "Home":
        home_page()
    elif page == "Quiz":
        quiz_page()
    else:
        analytics_page()

    render_footer()


if __name__ == "__main__":
    main()
