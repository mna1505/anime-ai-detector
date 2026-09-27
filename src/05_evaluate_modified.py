"""Evaluator modified untuk lima model dan empat level.

Label:
    0 = AI
    1 = Human

Level 4 menggunakan preprocessing DWT yang sama dengan training.
"""

# ============================================================
# DOKUMENTASI COMMAND — 05_evaluate_modified.py
# ============================================================
# FUNGSI:
#   Mengevaluasi checkpoint Level 1-4 pada:
#       - in_domain
#       - cross_generator
#
# FOKUS MODEL:
#   - mobilenetv3 = MobileNetV3-Large
#   - deit_tiny  = DeiT-Tiny
#
# FORMAT COMMAND:
#   python src/05_evaluate_modified.py --level <LEVEL> --model <MODEL>
#
# CONTOH PER LEVEL:
#   python src/05_evaluate_modified.py --level 1 --model mobilenetv3
#   python src/05_evaluate_modified.py --level 1 --model deit_tiny
#
#   python src/05_evaluate_modified.py --level 2 --model mobilenetv3
#   python src/05_evaluate_modified.py --level 2 --model deit_tiny
#
#   python src/05_evaluate_modified.py --level 3 --model mobilenetv3
#   python src/05_evaluate_modified.py --level 3 --model deit_tiny
#
#   python src/05_evaluate_modified.py --level 4 --model mobilenetv3
#   python src/05_evaluate_modified.py --level 4 --model deit_tiny
#
# MENJALANKAN KEDUA MODEL PADA SATU LEVEL:
#   python src/05_evaluate_modified.py --level 1 --all-models
#   python src/05_evaluate_modified.py --level 2 --all-models
#   python src/05_evaluate_modified.py --level 3 --all-models
#   python src/05_evaluate_modified.py --level 4 --all-models
#
# OUTPUT:
#   results_modified/evaluation_summary_<LEVEL>.csv
#   results_modified/evaluation_summary_<LEVEL>.json
#
# CATATAN:
#   Level 4 menggunakan preprocessing DWT yang sama dengan training.
# ============================================================


from pathlib import Path
from importlib import import_module
import argparse
import csv
import json
import os
import threading
import time

import psutil
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

model_factory = import_module("model_factory_modified")
preprocessing = import_module("02_preprocessing_modified")
dataset_module = import_module("03_dataset_modified")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_modified"
RESULT_DIR = ROOT_DIR / "results_modified"
MODELS = model_factory.MODELS
CONDITIONS = ("in_domain", "cross_generator")
BATCH_SIZE = 32
NUM_WORKERS = 2
WARMUP_BATCHES = 3
DEVICE = preprocessing.get_device()


class PeakMemoryMonitor:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = self.process.memory_info().rss
        self.running = False
        self.thread = None

    def _run(self):
        while self.running:
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    @property
    def peak_ram_mb(self):
        return self.peak_rss / (1024 ** 2)


def reset_gpu_peak_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_vram_mb():
    if DEVICE.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load_model(model_name, level):
    # Menyuntikkan parameter level agar factory otomatis mengonversi ke gerbang 4 channel untuk Level 4
    model = model_factory.create_model(model_name, pretrained=False, num_classes=2, level=level)
    checkpoint = MODEL_DIR / f"level{level}" / model_name / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint}. "
            f"Jalankan 04_train_modified.py --level {level} terlebih dahulu."
        )
    state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()



def evaluate_model(model_name, condition, level):
    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=model_name,
        level=level,
    )
    loader = loaders["test_in_domain"] if condition == "in_domain" else loaders["test_cross_generator"]
    model = load_model(model_name, level)

    # Warm-up tidak masuk pengukuran metrik maupun waktu utama.
    with torch.inference_mode():
        for i, (images, _) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            synchronize()
            _ = model(images)
            synchronize()
            if i + 1 >= WARMUP_BATCHES:
                break

    reset_gpu_peak_memory()
    ram_monitor = PeakMemoryMonitor()
    baseline_ram_mb = ram_monitor.process.memory_info().rss / (1024 ** 2)
    ram_monitor.start()

    y_true, y_pred, ai_prob = [], [], []
    total_forward_seconds = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            synchronize()
            start = time.perf_counter()
            outputs = model(images)
            synchronize()
            total_forward_seconds += time.perf_counter() - start

            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            ai_prob.extend(probabilities[:, 0].cpu().tolist())
            total_images += labels.size(0)

    ram_monitor.stop()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, ai_prob)
    except ValueError:
        roc_auc = float("nan")

    latency_ms = (total_forward_seconds / max(1, total_images)) * 1000.0
    throughput = total_images / total_forward_seconds if total_forward_seconds > 0 else float("inf")

    return {
        "model": model_name,
        "condition": condition,
        "level": level,
        "number_of_images": total_images,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "total_inference_time_seconds": total_forward_seconds,
        "inference_latency_ms_per_image": latency_ms,
        "throughput_images_per_second": throughput,
        "peak_gpu_vram_mb_inference": get_peak_gpu_vram_mb(),
        "baseline_ram_mb_inference": baseline_ram_mb,
        "peak_ram_mb_inference": ram_monitor.peak_ram_mb,
        "peak_ram_delta_mb_inference": ram_monitor.peak_ram_mb - baseline_ram_mb,
        "label_mapping": "0=AI, 1=Human",
    }


def save_results(rows, level):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"evaluation_summary_{level}.csv"
    fields = [
        "model", "condition", "number_of_images", "confusion_matrix",
        "accuracy", "precision", "recall", "f1_score", "roc_auc",
        "total_inference_time_seconds", "inference_latency_ms_per_image",
        "throughput_images_per_second", "peak_gpu_vram_mb_inference",
        "baseline_ram_mb_inference", "peak_ram_mb_inference",
        "peak_ram_delta_mb_inference",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fields}
            out["confusion_matrix"] = json.dumps(out["confusion_matrix"])
            writer.writerow(out)

    json_path = RESULT_DIR / f"evaluation_summary_{level}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate modified Anime AI detector")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--all-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = MODELS if args.all_models or args.model is None else (args.model,)
    rows = []
    for model_name in selected:
        for condition in CONDITIONS:
            print(f"Evaluating {model_name} | Level {args.level} | {condition}")
            result = evaluate_model(model_name, condition, args.level)
            rows.append(result)
            print(
                f"accuracy={result['accuracy']:.4f} | f1={result['f1_score']:.4f} | "
                f"latency={result['inference_latency_ms_per_image']:.3f} ms/image"
            )
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    path = save_results(rows, args.level)
    print(f"Hasil evaluasi disimpan: {path}")


if __name__ == "__main__":
    main()
