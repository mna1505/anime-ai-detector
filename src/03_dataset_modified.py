"""DataLoader modified untuk empat split penelitian.

Label wajib:
    ai    -> 0
    human -> 1
"""

# ============================================================
# DOKUMENTASI COMMAND — 03_dataset_modified.py
# ============================================================
# Fungsi utama:
#   Menyediakan DataLoader untuk pipeline Level 1 sampai Level 4.
#
# COMMAND UTAMA:
#   Dari ROOT proyek:
#       python src/03_dataset_modified.py
#
#   Dari folder src:
#       python 03_dataset_modified.py
#
# FOKUS MODEL PENELITIAN:
#   - mobilenetv3
#   - deit_tiny
#
# CATATAN:
#   File ini tidak menerima --model dari terminal.
#   Pemilihan model/level pada training dan evaluation dilakukan
#   melalui 04_train_modified.py dan 05_evaluate_modified.py.
# ============================================================


from pathlib import Path
from importlib import import_module
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "dataset" / "splits"
preprocessing = import_module("02_preprocessing_modified")

BATCH_SIZE_DEFAULT = 32
NUM_WORKERS_DEFAULT = 2
RANDOM_SEED = 42


def create_dataset(folder, transform):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
    dataset = ImageFolder(root=str(folder), transform=transform)
    expected = {"ai": 0, "human": 1}
    if dataset.class_to_idx != expected:
        raise RuntimeError(
            f"Mapping label tidak sesuai aturan penelitian. Diharapkan {expected}, "
            f"ditemukan {dataset.class_to_idx} pada {folder}"
        )
    return dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)


def get_dataloaders(batch_size=BATCH_SIZE_DEFAULT, num_workers=NUM_WORKERS_DEFAULT, model_name="resnet50", level=1):
    train_dataset = create_dataset(
        SPLIT_DIR / "train",
        preprocessing.get_train_transform(model_name=model_name, level=level),
    )
    validation_dataset = create_dataset(
        SPLIT_DIR / "validation",
        preprocessing.get_eval_transform(level=level),
    )
    test_in_domain_dataset = create_dataset(
        SPLIT_DIR / "test_in_domain",
        preprocessing.get_eval_transform(level=level),
    )
    test_cross_dataset = create_dataset(
        SPLIT_DIR / "test_cross_generator",
        preprocessing.get_eval_transform(level=level),
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(RANDOM_SEED)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(RANDOM_SEED)

    return {
        "train": DataLoader(train_dataset, shuffle=True, generator=train_generator, **common),
        "validation": DataLoader(validation_dataset, shuffle=False, generator=eval_generator, **common),
        "test_in_domain": DataLoader(test_in_domain_dataset, shuffle=False, generator=eval_generator, **common),
        "test_cross_generator": DataLoader(test_cross_dataset, shuffle=False, generator=eval_generator, **common),
    }


def print_dataset_information():
    for level in (1, 4):
        loaders = get_dataloaders(model_name="mobilenetv3", level=level)
        print(f"\n=== DATASET INFORMATION / LEVEL {level} ===")
        for name, loader in loaders.items():
            dataset = loader.dataset
            counts = {class_name: 0 for class_name in dataset.classes}
            for _, label in dataset.samples:
                counts[dataset.classes[label]] += 1
            print(f"{name}: {len(dataset)} | classes={dataset.classes} | mapping={dataset.class_to_idx} | distribution={counts}")


if __name__ == "__main__":
    print_dataset_information()
