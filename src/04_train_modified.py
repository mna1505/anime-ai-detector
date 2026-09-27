"""Training pipeline Baseline -> Level 1 -> Level 4 untuk lima arsitektur.

Perubahan utama dari codebase lama:
1. ViT-B/16 diganti menjadi DeiT-Base dan DeiT-Tiny.
2. Level 4 TIDAK menggunakan SAM.
3. Level 4 menggunakan preprocessing Haar DWT.
4. Level 1-3 mempertahankan strategi training lama:
   - L1: AdamW + cosine + augmentasi
   - L2: L1 + MixUp + Label Smoothing + warm-up
   - L3: L2 + discriminative learning rate
5. Label penelitian: 0=AI, 1=Human.
"""

# ============================================================
# DOKUMENTASI COMMAND — 04_train_modified.py
# ============================================================
# FUNGSI:
#   Menjalankan training Level 1, Level 2, Level 3, atau Level 4.
#
# FOKUS MODEL:
#   - mobilenetv3 = MobileNetV3-Large
#   - deit_tiny  = DeiT-Tiny
#
# FORMAT COMMAND:
#   python src/04_train_modified.py --level <LEVEL> --model <MODEL>
#
# MODEL MOBILE-NET:
#   python src/04_train_modified.py --level 1 --model mobilenetv3
#   python src/04_train_modified.py --level 2 --model mobilenetv3
#   python src/04_train_modified.py --level 3 --model mobilenetv3
#   python src/04_train_modified.py --level 4 --model mobilenetv3
#
# MODEL DEIT-TINY:
#   python src/04_train_modified.py --level 1 --model deit_tiny
#   python src/04_train_modified.py --level 2 --model deit_tiny
#   python src/04_train_modified.py --level 3 --model deit_tiny
#   python src/04_train_modified.py --level 4 --model deit_tiny
#
# MENJALANKAN KEDUA MODEL PADA SATU LEVEL:
#   python src/04_train_modified.py --level 1 --all-models
#   python src/04_train_modified.py --level 2 --all-models
#   python src/04_train_modified.py --level 3 --all-models
#   python src/04_train_modified.py --level 4 --all-models
#
# URUTAN LEVEL:
#   Level 1 = AdamW + cosine + augmentation
#   Level 2 = Level 1 + MixUp + Label Smoothing + warm-up
#   Level 3 = Level 2 + discriminative learning rate
#   Level 4 = DWT
#
# OUTPUT:
#   models_modified/level<LEVEL>/<model>/best_model.pth
#   results_modified/level<LEVEL>/...
#
# CATATAN:
#   Level 4 tidak menggunakan SAM.
# ============================================================


from pathlib import Path
from importlib import import_module
import argparse
import csv
import json
import os
import random
import threading
import time

import psutil
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

model_factory = import_module("model_factory_modified")
preprocessing = import_module("02_preprocessing_modified")
dataset_module = import_module("03_dataset_modified")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_modified"
RESULT_DIR = ROOT_DIR / "results_modified"

MODELS = model_factory.MODELS
BATCH_SIZE = 32
NUM_EPOCHS = 25
NUM_WORKERS = 2
RANDOM_SEED = 42
PRETRAINED = True
EARLY_STOPPING_PATIENCE = 6
BASE_LR = 1e-4
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.10
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.1
DEVICE = preprocessing.get_device()


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def create_model(model_name, pretrained=PRETRAINED):
    return model_factory.create_model(model_name, pretrained=pretrained, num_classes=2)


def count_parameters(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def configure_level3(model, model_name):
    backbone, head = model_factory.split_backbone_head_parameters(model, model_name)
    return [
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": head, "lr": HEAD_LR},
    ]


def build_optimizer_and_scheduler(model, model_name, level):
    if level == 1:
        optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif level == 2:
        optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    elif level in (3, 4):
        # Level 4 mempertahankan optimizer/scheduler Level 3.
        # SAM dihapus; DWT berada pada preprocessing Level 4.
        optimizer = AdamW(configure_level3(model, model_name), weight_decay=WEIGHT_DECAY)
        warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    else:
        raise ValueError("level harus 1, 2, 3, atau 4")
    return optimizer, scheduler


def mixup_data(images, labels, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1.0 - lam) * images[index], labels, labels[index], lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)


def train_one_epoch(model, loader, criterion, optimizer, level):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if level >= 2:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def get_current_lr(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_training_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train_model(model_name, level):
    set_seed(RANDOM_SEED)
    reset_gpu_peak_memory()
    output_dir = MODEL_DIR / f"level{level}" / model_name
    result_dir = RESULT_DIR / f"level{level}" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=model_name,
        level=level,
    )

    # model = create_model(model_name, pretrained=PRETRAINED).to(DEVICE)
    model = model_factory.create_model(model_name, pretrained=PRETRAINED, num_classes=2, level=level).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=(LABEL_SMOOTHING if level >= 2 else 0.0))
    optimizer, scheduler = build_optimizer_and_scheduler(model, model_name, level)

    total_params, trainable_params = count_parameters(model)
    memory_monitor = PeakMemoryMonitor()
    memory_monitor.start()
    start = time.perf_counter()

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, level)
        val_loss, val_acc = validate(model, loaders["validation"], criterion)
        scheduler.step()
        lrs = get_current_lr(optimizer)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate_backbone": lrs[0],
            "learning_rate_head": lrs[-1],
        }
        history.append(row)
        print(
            f"[{model_name}] Level {level} Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping pada epoch {epoch}.")
                break

    elapsed = time.perf_counter() - start
    memory_monitor.stop()

    summary = {
        "model": model_name,
        "model_display_name": model_factory.MODEL_DISPLAY_NAMES[model_name],
        "level": level,
        "level_strategy": {
            "level_1": "AdamW + cosine + robustness augmentation",
            "level_2": "Level 1 + MixUp + label smoothing + warm-up",
            "level_3": "Level 2 + discriminative learning rate",
            "level_4": "Level 3 optimizer policy + Haar DWT preprocessing; SAM removed",
        }[f"level_{level}"],
        "label_mapping": {"AI": 0, "Human": 1},
        "best_validation_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "total_training_time_seconds": elapsed,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "peak_gpu_vram_mb_training": get_peak_gpu_vram_mb(),
        "peak_ram_mb_training": memory_monitor.peak_ram_mb,
        "checkpoint": str(output_dir / "best_model.pth"),
    }

    save_training_history(history, result_dir / "training_history.csv")
    save_json(summary, result_dir / "training_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Training modified Anime AI detector")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--all-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = MODELS if args.all_models or args.model is None else (args.model,)
    
    # 1. Jalankan pelatihan untuk model yang dipilih
    summaries = [train_model(model_name, args.level) for model_name in selected]
    
    # 2. Ambil jalur folder induk level saat ini (misal: results_modified/level1/)
    level_result_dir = RESULT_DIR / f"level{args.level}"
    level_result_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Ekspor akumulasi ke file JSON gabungan
    with open(level_result_dir / "training_efficiency_summary.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)
        
    # 4. Ekspor akumulasi ke file CSV gabungan terpusat
    if summaries:
        fields = [
            "model", "model_display_name", "level", "best_validation_accuracy",
            "best_epoch", "epochs_completed", "total_training_time_seconds",
            "total_parameters", "trainable_parameters", "peak_gpu_vram_mb_training",
            "peak_ram_mb_training"
        ]
        
        with open(level_result_dir / "training_efficiency_summary.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row.get(key) for key in fields})
                
    print(f"\n[SUKSES] Berkas ringkasan efisiensi Level {args.level} berhasil disimpan di: {level_result_dir}")


if __name__ == "__main__":
    main()

