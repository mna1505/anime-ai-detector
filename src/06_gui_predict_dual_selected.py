"""
===============================================================================
06_gui_predict_dual_selected.py
===============================================================================
GUI Gradio untuk deployment dua checkpoint hasil eksperimen terpilih.

Pilihan mode pada GUI:
-------------------------------------------------------------------------------
1. MobileNetV3-Large — Level 4 (Haar DWT)
2. DeiT-Tiny        — Level 3 (Discriminative Learning Rate)
3. MobileNetV3-Large + DeiT-Tiny secara bersamaan

Perilaku memori:
-------------------------------------------------------------------------------
- Mode 1 hanya memuat checkpoint MobileNetV3-Large Level 4.
- Mode 2 hanya memuat checkpoint DeiT-Tiny Level 3.
- Mode 3 memuat kedua checkpoint agar pengguna dapat memperoleh dua prediksi
  dalam satu proses deteksi.
- Lazy loading tetap digunakan: model baru dimuat ketika tombol Deteksi ditekan.
- Saat berpindah dari mode 3 ke mode 1/2, kedua model dilepas terlebih dahulu.
- Tombol pembersihan memori dapat digunakan kapan saja.

Checkpoint yang digunakan:
-------------------------------------------------------------------------------
models_modified/
├── level4/
│   └── mobilenetv3/
│       └── best_model.pth
└── level3/
    └── deit_tiny/
        └── best_model.pth

Catatan:
-------------------------------------------------------------------------------
- MobileNetV3-Large menggunakan preprocessing Level 4 dengan Haar DWT,
  sehingga input model memiliki 4 channel (RGB + DWT).
- DeiT-Tiny menggunakan preprocessing Level 3 standar modified,
  sehingga input model memiliki 3 channel RGB.
- Label dataset:
      0 = AI-generated
      1 = Human-made
- Bagian "alasan prediksi" menjelaskan indikasi keputusan berdasarkan
  probabilitas keluaran model. Bagian tersebut bukan penjelasan kausal
  feature-level seperti Grad-CAM atau SHAP.
- Grafik statistik pada GUI menggunakan HTML/CSS sederhana dengan animasi
  transisi ringan; tidak menggunakan plotting engine berat.
- File ini tidak melakukan training ulang.
- GUI tidak menggunakan emoji sebagai icon.
===============================================================================
"""

import gc
import html
import time
from importlib import import_module
from pathlib import Path

import gradio as gr
import torch
from PIL import Image


# ============================================================
# INISIALISASI MODUL DAN DEVICE
# ============================================================
model_factory = import_module("model_factory_modified")
preprocessing = import_module("02_preprocessing_modified")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_modified"
DEVICE = preprocessing.get_device()


# ============================================================
# KONFIGURASI CHECKPOINT TERPILIH
# ============================================================
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


# ============================================================
# MODEL CACHE
# ============================================================
# Mode 1/2 menggunakan satu model. Mode 3 sengaja dapat menyimpan dua model
# karena pengguna memang meminta perbandingan dua prediksi secara cepat.
CURRENT_LOADED_MODELS = {}


def get_selected_config(config_key):
    """Mengembalikan konfigurasi checkpoint berdasarkan pilihan GUI."""
    if config_key not in SELECTED_MODEL_CONFIGS:
        raise ValueError(f"Pilihan model tidak dikenal: {config_key}")
    return SELECTED_MODEL_CONFIGS[config_key]


def get_checkpoint_path(config_key):
    """Mengembalikan path best_model.pth sesuai konfigurasi checkpoint."""
    config = get_selected_config(config_key)
    return (
        MODEL_DIR
        / f"level{config['level']}"
        / config["model_key"]
        / "best_model.pth"
    )


def release_models():
    """Melepaskan seluruh model aktif dari RAM/VRAM."""
    global CURRENT_LOADED_MODELS

    CURRENT_LOADED_MODELS.clear()
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def load_model(config_key):
    """Lazy-load satu checkpoint dan mengembalikan model serta statusnya."""
    if config_key not in SELECTED_MODEL_CONFIGS:
        raise ValueError(f"Pilihan model tidak valid: {config_key}")

    if config_key in CURRENT_LOADED_MODELS:
        return CURRENT_LOADED_MODELS[config_key]

    config = get_selected_config(config_key)
    checkpoint = get_checkpoint_path(config_key)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint}"
        )

    model = None

    try:
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

    except Exception:
        if model is not None:
            del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise


def prepare_input(config, input_image):
    """Menerapkan transform evaluasi sesuai level checkpoint."""
    transform = preprocessing.get_eval_transform(level=config["level"])

    if isinstance(input_image, Image.Image):
        pil_image = input_image.convert("RGB")
    else:
        pil_image = Image.fromarray(input_image).convert("RGB")

    return transform(pil_image).unsqueeze(0).to(DEVICE)


# ============================================================
# KOMPONEN STATISTIK ANIMASI RINGAN
# ============================================================
def build_prediction_card(result):
    """
    Membuat kartu statistik berbasis HTML/CSS.

    Grafik hanya menggunakan div + CSS transition sehingga jauh lebih ringan
    daripada membuat figure matplotlib pada setiap prediksi.
    """
    config = result["config"]
    ai_prob = result["ai_probability"]
    human_prob = result["human_probability"]
    latency = result["latency_ms"]
    prediction = result["label"]

    ai_width = max(0.0, min(100.0, ai_prob))
    human_width = max(0.0, min(100.0, human_prob))
    latency_scale = max(4.0, min(100.0, latency * 8.0))

    safe_name = html.escape(config["display_name"])
    safe_stage = html.escape(config["stage_name"])
    safe_strategy = html.escape(config["strategy"])
    safe_prediction = html.escape(prediction)
    safe_input = html.escape(config["input_description"])

    if prediction == "AI-generated":
        reason = (
            "Probabilitas AI-generated lebih tinggi daripada Human-made. "
            "Artinya, pola fitur visual yang diterima jaringan lebih dekat "
            "dengan pola yang dipelajari sebagai citra AI-generated pada data training."
        )
    else:
        reason = (
            "Probabilitas Human-made lebih tinggi daripada AI-generated. "
            "Artinya, pola fitur visual yang diterima jaringan lebih dekat "
            "dengan pola yang dipelajari sebagai citra Human-made pada data training."
        )

    return f"""
<div class="model-card">
  <div class="model-card-header">
    <div>
      <div class="model-name">{safe_name}</div>
      <div class="model-stage">{safe_stage} &middot; {safe_strategy}</div>
    </div>
    <div class="prediction-badge">{safe_prediction}</div>
  </div>

  <div class="reason-box">
    <div class="section-title">Mengapa model memberikan prediksi ini?</div>
    <div class="reason-text">{html.escape(reason)}</div>
  </div>

  <div class="stats-grid">
    <div class="stat-panel">
      <div class="section-title">Probabilitas Kelas</div>
      <div class="bar-row">
        <div class="bar-label"><span>AI-generated</span><strong>{ai_prob:.2f}%</strong></div>
        <div class="bar-track"><div class="bar-fill ai-bar" style="width:{ai_width:.2f}%"></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>Human-made</span><strong>{human_prob:.2f}%</strong></div>
        <div class="bar-track"><div class="bar-fill human-bar" style="width:{human_width:.2f}%"></div></div>
      </div>
    </div>

    <div class="stat-panel latency-panel">
      <div class="section-title">Inference Latency</div>
      <div class="latency-value">{latency:.2f} <span>ms</span></div>
      <div class="latency-track"><div class="latency-fill" style="width:{latency_scale:.2f}%"></div></div>
      <div class="latency-note">Waktu inferensi satu gambar pada {html.escape(str(DEVICE))}</div>
    </div>
  </div>

</div>
"""


def build_results_html(results):
    """Menggabungkan satu atau dua kartu prediksi menjadi tampilan GUI."""
    if not results:
        return "<div class='empty-state'>Belum ada hasil prediksi.</div>"

    cards = "".join(build_prediction_card(result) for result in results)
    count_text = "2 model aktif untuk perbandingan." if len(results) == 2 else "1 model aktif."

    return f"""
<div class="results-wrapper">
  <div class="results-heading">Hasil Analisis</div>
  <div class="results-subheading">{count_text}</div>
  {cards}
</div>
"""


# ============================================================
# PREPROCESSING DAN PREDIKSI
# ============================================================
def run_single_prediction(config_key, input_image):
    """Menjalankan inferensi satu checkpoint dan mengembalikan statistiknya."""
    config = get_selected_config(config_key)
    model = load_model(config_key)
    tensor = prepare_input(config, input_image)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction = int(outputs.argmax(dim=1).item())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "config_key": config_key,
        "config": config,
        "label": "AI-generated" if prediction == 0 else "Human-made",
        "ai_probability": float(probabilities[0].item() * 100.0),
        "human_probability": float(probabilities[1].item() * 100.0),
        "latency_ms": elapsed_ms,
    }


def predict_image(mode_key, input_image):
    """Menjalankan mode satu model atau dua model sesuai pilihan pengguna."""
    if input_image is None:
        return (
            "<div class='empty-state'>Silakan unggah gambar terlebih dahulu.</div>",
            "Belum ada gambar yang diproses.",
        )

    try:
        if mode_key == "both":
            selected_keys = ["mobilenetv3_l4", "deit_tiny_l3"]
        elif mode_key in SELECTED_MODEL_CONFIGS:
            selected_keys = [mode_key]
        else:
            raise ValueError(f"Mode tidak dikenal: {mode_key}")

        # Mode 1/2 hanya membutuhkan satu checkpoint.
        # Mode 3 sengaja mempertahankan dua checkpoint agar dua prediksi dapat
        # diberikan dari satu klik tombol.
        selected_set = set(selected_keys)
        stale_keys = set(CURRENT_LOADED_MODELS) - selected_set
        if stale_keys:
            for stale_key in stale_keys:
                del CURRENT_LOADED_MODELS[stale_key]
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        results = [
            run_single_prediction(config_key, input_image)
            for config_key in selected_keys
        ]

        status = (
            "Mode dua model: MobileNetV3-Large Level 4 dan DeiT-Tiny Level 3 "
            "aktif untuk perbandingan."
            if mode_key == "both"
            else f"{get_selected_config(mode_key)['display_name']} "
                 f"{get_selected_config(mode_key)['stage_name']} aktif."
        )

        return build_results_html(results), status

    except Exception as exc:
        release_models()
        return (
            f"<div class='error-state'><b>Inferensi gagal.</b><br>{html.escape(str(exc))}</div>",
            f"Error: {exc}",
        )


def on_mode_change(mode_key):
    """Melepas model yang tidak diperlukan ketika mode GUI diganti."""
    if mode_key == "both":
        wanted = {"mobilenetv3_l4", "deit_tiny_l3"}
        description = (
            "Mode perbandingan dua model dipilih. Kedua checkpoint akan dimuat "
            "saat tombol Deteksi ditekan."
        )
    else:
        wanted = {mode_key}
        config = get_selected_config(mode_key)
        description = (
            f"{config['display_name']} — {config['stage_name']} dipilih. "
            "Checkpoint akan dimuat secara lazy saat tombol Deteksi ditekan."
        )

    stale_keys = set(CURRENT_LOADED_MODELS) - wanted
    if stale_keys:
        for stale_key in stale_keys:
            del CURRENT_LOADED_MODELS[stale_key]
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return description


def clear_interface():
    """Membersihkan input/output sekaligus melepaskan semua model."""
    release_models()

    return (
        None,
        "<div class='empty-state'>Model dilepas dari memori.</div>",
        "Semua checkpoint dilepas dari RAM/VRAM.",
    )


# ============================================================
# DESAIN ANTARMUKA WEB
# ============================================================
CUSTOM_CSS = """
:root {
  --page-bg: #101216;
  --card-bg: #22262c;
  --panel-bg: #2a2f36;
  --panel-bg-soft: #272c32;
  --border: #3a414b;
  --text: #f5f7fa;
  --text-soft: #e2e6eb;
  --muted: #b6bec9;
  --accent: #4f8cff;
  --accent-soft: #263b63;
  --accent-text: #bcd2ff;
  --track: #3b424c;
  --human: #8b98aa;
  --latency: #3fa7a0;
}

/* ============================================================
   TEMA GELAP GUI
   ============================================================ */
.gradio-container {
  background: var(--page-bg) !important;
  color: var(--text) !important;
}

.gradio-container .prose,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong {
  color: var(--text) !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .block-label,
.gradio-container .block-label span {
  color: var(--text-soft) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #1b1e23 !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #89929e !important;
}

.gradio-container button:not(.primary) {
  background: #2a2f36 !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}

.model-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  margin: 14px 0;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
  animation: cardIn 260ms ease-out;
}

@keyframes cardIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.model-name {
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
}

.model-stage {
  color: var(--muted);
  font-size: 13px;
  margin-top: 4px;
}

.prediction-badge {
  border: 1px solid #385d9b;
  background: #1e3357;
  color: var(--accent-text);
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 700;
  white-space: nowrap;
}

.reason-box,
.stat-panel {
  background: var(--panel-bg);
  border: 1px solid #343b45;
  border-radius: 12px;
  padding: 15px;
}

.reason-box {
  margin-bottom: 14px;
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: var(--muted);
  margin-bottom: 8px;
}

.reason-text {
  color: var(--text-soft);
  line-height: 1.55;
  font-size: 14px;
}

.stats-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(220px, .8fr);
  gap: 14px;
}

.bar-row {
  margin: 12px 0;
}

.bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--text-soft);
  margin-bottom: 6px;
}

.bar-label strong {
  color: var(--text);
}

.bar-track,
.latency-track {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--track);
}

.bar-fill,
.latency-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 700ms cubic-bezier(.2,.8,.2,1);
}

.ai-bar {
  background: #4f8cff;
}

.human-bar {
  background: var(--human);
}

.latency-fill {
  background: var(--latency);
}

.latency-value {
  font-size: 30px;
  font-weight: 750;
  color: var(--text);
  margin: 8px 0 14px;
}

.latency-value span {
  font-size: 14px;
  font-weight: 600;
  color: var(--muted);
}

.latency-note {
  margin-top: 9px;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.4;
}

.model-meta {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--muted);
  word-break: break-word;
}

.model-meta b {
  color: var(--text-soft);
}

.results-heading {
  font-size: 22px;
  font-weight: 750;
  color: var(--text) !important;
  margin-top: 4px;
}

.results-subheading {
  font-size: 13px;
  color: var(--muted) !important;
  margin: 4px 0 8px;
}

.empty-state,
.error-state {
  padding: 18px;
  border: 1px dashed var(--border);
  border-radius: 12px;
  color: var(--muted);
  background: var(--panel-bg);
}

.error-state {
  color: #ffb4ab;
}

@media (max-width: 800px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .model-card-header {
    align-items: flex-start;
    flex-direction: column;
  }
}
"""

with gr.Blocks(
    title="Anime AI Illustration Detector",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
# Anime AI vs Human-Made Illustration Detector

Pilih cara analisis yang diinginkan:

- **MobileNetV3-Large** — Level 4 (Haar DWT)
- **DeiT-Tiny** — Level 3 (Discriminative Learning Rate)
- **Keduanya** — menjalankan dua checkpoint untuk perbandingan cepat

Mode satu model tetap hemat memori. Mode **Keduanya** sengaja memuat dua
checkpoint agar hasil dua model dapat diperoleh dalam satu klik.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=[
                    ("MobileNetV3-Large — Level 4 (Haar DWT)", "mobilenetv3_l4"),
                    ("DeiT-Tiny — Level 3 (Discriminative Learning Rate)", "deit_tiny_l3"),
                    ("MobileNetV3-Large + DeiT-Tiny", "both"),
                ],
                value="mobilenetv3_l4",
                label="Mode Analisis",
            )

            image_input = gr.Image(
                label="Unggah atau Tarik Gambar Anime",
                type="numpy",
            )

            with gr.Row():
                detect_btn = gr.Button("Deteksi", variant="primary")
                clear_btn = gr.Button("Bersihkan dan Lepas Model")

            model_status = gr.Textbox(
                label="Status Model",
                value=on_mode_change("mobilenetv3_l4"),
                interactive=False,
            )

        with gr.Column(scale=1):
            results_html = gr.HTML(
                value="<div class='empty-state'>Belum ada hasil prediksi.</div>",
                label="Statistik Prediksi",
            )

    model_dropdown.change(
        fn=on_mode_change,
        inputs=[model_dropdown],
        outputs=[model_status],
    )

    detect_btn.click(
        fn=predict_image,
        inputs=[model_dropdown, image_input],
        outputs=[results_html, model_status],
    )

    clear_btn.click(
        fn=clear_interface,
        inputs=[],
        outputs=[image_input, results_html, model_status],
    )


if __name__ == "__main__":
    demo.launch()
