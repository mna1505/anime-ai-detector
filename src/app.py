
"""
Streamlit GUI untuk Anime AI vs Human-Made Illustration Detector.

Model:
- MobileNetV3-Large — Level 4 (Haar DWT)
- DeiT-Tiny — Level 3 (Discriminative Learning Rate)

Deployment:
- CPU-friendly inference
- Lazy loading: checkpoint hanya dimuat saat diperlukan
- Bisa menjalankan satu model atau kedua model
"""

import gc
import html
import threading
import time
from importlib import import_module
from pathlib import Path

import streamlit as st
import torch
from PIL import Image


# ============================================================
# PROJECT / MODEL CONFIG
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models_modified"

model_factory = import_module("model_factory_modified")
preprocessing = import_module("02_preprocessing_modified")

DEVICE = torch.device("cpu")

SELECTED_MODEL_CONFIGS = {
    "mobilenetv3_l4": {
        "model_key": "mobilenetv3",
        "display_name": "MobileNetV3-Large",
        "level": 4,
        "stage_name": "Level 4",
        "strategy": "Haar DWT",
        "input_description": "4 channel (RGB + DWT)",
    },
    "deit_tiny_l3": {
        "model_key": "deit_tiny",
        "display_name": "DeiT-Tiny",
        "level": 3,
        "stage_name": "Level 3",
        "strategy": "Discriminative Learning Rate",
        "input_description": "3 channel RGB",
    },
}

# Global cache per Streamlit process.
# CPU inference tidak perlu CUDA dan model hanya dimuat ketika dipakai.
CURRENT_LOADED_MODELS = {}
MODEL_LOCK = threading.Lock()


def get_selected_config(config_key):
    if config_key not in SELECTED_MODEL_CONFIGS:
        raise ValueError(f"Pilihan model tidak dikenal: {config_key}")
    return SELECTED_MODEL_CONFIGS[config_key]


def get_checkpoint_path(config_key):
    config = get_selected_config(config_key)
    return (
        MODEL_DIR
        / f"level{config['level']}"
        / config["model_key"]
        / "best_model.pth"
    )


def release_models(keys=None):
    """Melepas model tertentu atau seluruh model dari RAM."""
    global CURRENT_LOADED_MODELS

    with MODEL_LOCK:
        if keys is None:
            CURRENT_LOADED_MODELS.clear()
        else:
            for key in keys:
                CURRENT_LOADED_MODELS.pop(key, None)

    gc.collect()


def load_model(config_key):
    """Lazy-load checkpoint ke CPU."""
    if config_key in CURRENT_LOADED_MODELS:
        return CURRENT_LOADED_MODELS[config_key]

    with MODEL_LOCK:
        if config_key in CURRENT_LOADED_MODELS:
            return CURRENT_LOADED_MODELS[config_key]

        config = get_selected_config(config_key)
        checkpoint = get_checkpoint_path(config_key)

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {checkpoint}\n"
                "Pastikan file best_model.pth berada pada struktur folder "
                "yang benar."
            )

        model = model_factory.create_model(
            config["model_key"],
            pretrained=False,
            num_classes=2,
            level=config["level"],
        )

        state_dict = torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()
        CURRENT_LOADED_MODELS[config_key] = model

        return model


def prepare_input(config, input_image):
    transform = preprocessing.get_eval_transform(level=config["level"])

    if isinstance(input_image, Image.Image):
        pil_image = input_image.convert("RGB")
    else:
        pil_image = Image.fromarray(input_image).convert("RGB")

    return transform(pil_image).unsqueeze(0).to(DEVICE)


def run_single_prediction(config_key, input_image):
    config = get_selected_config(config_key)
    model = load_model(config_key)
    tensor = prepare_input(config, input_image)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction = int(outputs.argmax(dim=1).item())

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "config_key": config_key,
        "config": config,
        "label": "AI-generated" if prediction == 0 else "Human-made",
        "ai_probability": float(probabilities[0].item() * 100.0),
        "human_probability": float(probabilities[1].item() * 100.0),
        "latency_ms": elapsed_ms,
    }


def render_prediction_card(result):
    """
    Render hasil model menggunakan komponen native Streamlit.
    Tidak memakai HTML card besar agar probability/statistics tidak
    berubah menjadi literal HTML/code block.
    """
    config = result["config"]
    ai_prob = float(result["ai_probability"])
    human_prob = float(result["human_probability"])
    latency = float(result["latency_ms"])
    prediction = result["label"]

    reason = (
        "Probabilitas AI-generated lebih tinggi daripada Human-made. "
        "Pola fitur visual yang diterima jaringan lebih dekat dengan pola "
        "yang dipelajari sebagai citra AI-generated pada data training."
        if prediction == "AI-generated"
        else
        "Probabilitas Human-made lebih tinggi daripada AI-generated. "
        "Pola fitur visual yang diterima jaringan lebih dekat dengan pola "
        "yang dipelajari sebagai citra Human-made pada data training."
    )

    # Card utama. st.container(border=True) dirender native oleh Streamlit.
    with st.container(border=True):
        header_left, header_right = st.columns([3.5, 1.2], gap="small")

        with header_left:
            st.markdown(
                f"### {config['display_name']}"
            )
            st.caption(
                f"{config['stage_name']} · {config['strategy']}"
            )

        with header_right:
            st.markdown(
                f'<div class="prediction-badge-native">{html.escape(prediction)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
**MENGAPA MODEL MEMBERIKAN PREDIKSI INI?**

{reason}
"""
        )

        # Statistik utama: dua probabilitas + latency.
        prob_col, latency_col = st.columns([1.7, 1.0], gap="medium")

        with prob_col:
            st.markdown("**PROBABILITAS KELAS**")

            ai_label_col, ai_value_col = st.columns([4, 1], gap="small")
            with ai_label_col:
                st.caption("AI-generated")
            with ai_value_col:
                st.markdown(
                    f'<div class="stat-value-right">{ai_prob:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            st.progress(max(0.0, min(1.0, ai_prob / 100.0)))

            human_label_col, human_value_col = st.columns([4, 1], gap="small")
            with human_label_col:
                st.caption("Human-made")
            with human_value_col:
                st.markdown(
                    f'<div class="stat-value-right">{human_prob:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            st.progress(max(0.0, min(1.0, human_prob / 100.0)))

        with latency_col:
            st.markdown("**INFERENCE LATENCY**")
            st.metric("Waktu inferensi", f"{latency:.2f} ms")
            st.caption("Waktu inferensi satu gambar pada CPU.")

        st.divider()

        meta_left, meta_right = st.columns(2, gap="small")
        with meta_left:
            st.caption(f"**Input:** {config['input_description']}")
        with meta_right:
            st.caption("**Checkpoint:** best_model.pth")


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Anime AI vs Human-Made Illustration Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The original Gradio interface is a two-column dashboard.  This CSS keeps
# the same visual hierarchy while using native Streamlit widgets for upload
# and buttons.
st.markdown(
    """
<style>
/* ---------- Page ---------- */
.stApp {
    background: #0f1115;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

/* Hide Streamlit chrome that is not part of the app UI. */
[data-testid="stSidebarNav"] { display: none; }

/* ---------- Typography ---------- */
.app-title {
    color: #f5f7fa;
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 .35rem 0;
}
.app-subtitle {
    color: #b9c0ca;
    font-size: .94rem;
    margin-bottom: 1.2rem;
}
.section-heading {
    color: #f5f7fa;
    font-size: 1.15rem;
    font-weight: 750;
    margin: 0 0 .55rem 0;
}

/* ---------- Gradio-like panels ---------- */
.ui-panel {
    background: #1d2026;
    border: 1px solid #343941;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}
.ui-panel-title {
    color: #dfe4eb;
    font-size: .86rem;
    font-weight: 650;
    margin-bottom: 9px;
}
.mode-help {
    color: #9fa7b2;
    font-size: .77rem;
    line-height: 1.45;
    margin-top: 7px;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    background: #1d2026;
    border: 1px solid #343941;
    border-radius: 10px;
    padding: 6px 10px 2px 10px;
}
[data-testid="stFileUploaderDropzone"] {
    background: #20242b;
    border: 1px dashed #4a515c;
    min-height: 72px;
}

/* ---------- Image frame ---------- */
.image-panel {
    background: #1d2026;
    border: 1px solid #343941;
    border-radius: 10px;
    padding: 10px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 7px;
    min-height: 42px;
    font-weight: 650;
}

/* ---------- Results / native Streamlit stats ---------- */
.results-title {
    color: #f5f7fa;
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0;
}
.results-subtitle {
    color: #9fa7b2;
    font-size: .8rem;
    margin: 2px 0 10px 0;
}
.prediction-badge-native {
    display: inline-block;
    float: right;
    color: #cfe0ff;
    background: #19365f;
    border: 1px solid #3565a5;
    border-radius: 999px;
    padding: 6px 11px;
    font-size: .72rem;
    font-weight: 750;
    white-space: nowrap;
    margin-top: 4px;
}
.stat-value-right {
    text-align: right;
    color: #f5f7fa;
    font-weight: 750;
    font-size: .82rem;
}
div[data-testid="stMetric"] {
    background: #282d34;
    border: 1px solid #353c46;
    border-radius: 10px;
    padding: 10px 12px;
}
div[data-testid="stMetricLabel"] {
    color: #aeb6c2;
}
div[data-testid="stMetricValue"] {
    color: #f5f7fa;
}
@media (max-width: 900px) {
    .stats-grid { grid-template-columns: 1fr; }
    .model-card-header { flex-direction: column; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    '<div class="app-title">Anime AI vs Human-Made Illustration Detector</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Upload ilustrasi anime untuk membandingkan prediksi MobileNetV3-Large dan DeiT-Tiny. Inference berjalan menggunakan CPU.</div>',
    unsafe_allow_html=True,
)

# ---------- Mode panel ----------
st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
st.markdown('<div class="ui-panel-title">Mode Analisis</div>', unsafe_allow_html=True)
mode = st.radio(
    "Pilih cara analisis yang diinginkan:",
    options=["mobilenetv3_l4", "deit_tiny_l3", "both"],
    format_func=lambda x: {
        "mobilenetv3_l4": "MobileNetV3-Large — Level 4 (Haar DWT)",
        "deit_tiny_l3": "DeiT-Tiny — Level 3 (Discriminative Learning Rate)",
        "both": "Keduanya — jalankan dua checkpoint untuk perbandingan cepat",
    }[x],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown(
    '<div class="mode-help">Mode satu model tetap hemat memori. Mode <b>Keduanya</b> memuat dua checkpoint agar hasil kedua model dapat diperoleh dalam satu klik.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Unggah atau Tarik Gambar Anime",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="visible",
)

# Persist the last uploaded image through Streamlit reruns.
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["current_image"] = image
else:
    image = st.session_state.get("current_image")

# ---------- Main two-column dashboard ----------
left_col, right_col = st.columns([1.02, 1.0], gap="large")

with left_col:
    st.markdown('<div class="section-heading">Gambar Input</div>', unsafe_allow_html=True)

    if image is not None:
        st.markdown('<div class="image-panel">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="image-panel"><div style="height:390px;display:flex;align-items:center;justify-content:center;color:#858e9b;font-size:.85rem;">Belum ada gambar. Silakan unggah ilustrasi anime.</div></div>',
            unsafe_allow_html=True,
        )

    button_col1, button_col2 = st.columns([1, 1], gap="small")
    with button_col1:
        detect_clicked = st.button("Deteksi", type="primary", use_container_width=True)
    with button_col2:
        release_clicked = st.button("Bersihkan dan Lepas Model", use_container_width=True)

    if release_clicked:
        release_models()
        st.session_state.pop("results", None)
        st.session_state["status"] = "Semua model telah dilepas dari RAM."
        st.rerun()

    st.markdown('<div class="ui-panel" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown('<div class="ui-panel-title">Status Model</div>', unsafe_allow_html=True)
    if mode == "both":
        status = "Mode dua model: MobileNetV3-Large Level 4 dan DeiT-Tiny Level 3 aktif untuk perbandingan."
    else:
        cfg = get_selected_config(mode)
        status = f"Mode satu model: {cfg['display_name']} {cfg['stage_name']} aktif."
    if "status" in st.session_state:
        status = st.session_state["status"]
    st.markdown(f'<div class="status-text">{html.escape(status)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="results-title">Hasil Analisis</div>', unsafe_allow_html=True)

    existing_results = st.session_state.get("results")
    if existing_results:
        st.markdown(
            f'<div class="results-subtitle">{len(existing_results)} model aktif untuk perbandingan.</div>',
            unsafe_allow_html=True,
        )
        # Render cards with native Streamlit components.
        # This avoids Streamlit interpreting the card as a Markdown code block.
        for result in existing_results:
            render_prediction_card(result)
    else:
        st.markdown(
            '<div class="ui-panel"><div class="mode-help">Belum ada hasil. Unggah gambar di sebelah kiri lalu tekan <b>Deteksi</b>.</div></div>',
            unsafe_allow_html=True,
        )

# ---------- Detection action ----------
if detect_clicked:
    if image is None:
        st.warning("Silakan unggah gambar terlebih dahulu.")
        st.stop()

    if mode == "both":
        selected_keys = ["mobilenetv3_l4", "deit_tiny_l3"]
    else:
        selected_keys = [mode]

    stale_keys = set(CURRENT_LOADED_MODELS) - set(selected_keys)
    if stale_keys:
        release_models(stale_keys)

    results = []
    progress = st.progress(0, text="Menyiapkan model...")

    try:
        for index, config_key in enumerate(selected_keys):
            config = get_selected_config(config_key)
            progress.progress(
                int(index / len(selected_keys) * 100),
                text=f"Menjalankan {config['display_name']}...",
            )
            results.append(run_single_prediction(config_key, image))

        progress.progress(100, text="Analisis selesai.")
        st.session_state["results"] = results
        st.session_state["status"] = (
            f"{len(results)} model aktif. Analisis terakhir berhasil dijalankan pada CPU."
        )
        time.sleep(0.15)
        st.rerun()

    except Exception as exc:
        st.session_state.pop("results", None)
        st.error(f"Inferensi gagal: {exc}")
