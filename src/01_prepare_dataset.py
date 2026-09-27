
# ============================================================
# DOKUMENTASI COMMAND — 01_prepare_dataset.py
# ============================================================
# Fungsi umum:
#   Menyiapkan/membangun dataset awal sesuai alur preprocessing dataset.
#
# COMMAND UTAMA:
#   Dari ROOT proyek:
#       python src/01_prepare_dataset.py
#
#   Dari folder src:
#       python 01_prepare_dataset.py
#
# CATATAN:
#   Script ini tidak menggunakan argparse, sehingga tidak memiliki
#   opsi --model atau --level.
#   Jalankan setelah sumber dataset dan struktur folder yang dibutuhkan
#   sudah tersedia.
#
# URUTAN:
#   1. Siapkan dataset mentah.
#   2. Jalankan script ini.
#   3. Periksa folder/output yang dibuat.
#   4. Lanjutkan ke 02_preprocessing.py.
# ============================================================

from pathlib import Path
import random
import shutil

from PIL import Image


# ============================================================
# ROOT DIRECTORY
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "dataset" / "raw"
PREPARED_DIR = ROOT_DIR / "dataset" / "prepared"
SPLIT_DIR = ROOT_DIR / "dataset" / "splits"


# ============================================================
# PEMBAGIAN JUMLAH GAMBAR
# KUOTA DIKUNCI MATI
# ============================================================
HUMAN_ANIMEDL2M_COUNT = 3250
AI_IN_DOMAIN_TOTAL = 3250

FLUX_COUNT = 1083
SDXL_COUNT = 1083
SD_COUNT = 1084

DANBOORU_COUNT = 500
PIXIV_COUNT = 500

NOVELAI_TOTAL = 500
NOVELAI_832_COUNT = 167
NOVELAI_1024_COUNT = 167
NOVELAI_1216_COUNT = 166


# ============================================================
# PEMBAGIAN TRAIN / VALIDATION / TEST
# ============================================================
TRAIN_PER_CLASS = 2250
VAL_PER_CLASS = 500
TEST_IN_DOMAIN_PER_CLASS = 500

RANDOM_SEED = 42


# ============================================================
# BATASAN DIMENSI GAMBAR
# ============================================================
MIN_WIDTH = 512
MIN_HEIGHT = 512

MAX_WIDTH = 2048
MAX_HEIGHT = 2048

MAX_ASPECT_RATIO = 2.0


# ============================================================
# EKSTENSI GAMBAR YANG DIDUKUNG
# ============================================================
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# DIRECTORY PATHS
# ============================================================
ANIMEDL2M_REAL = RAW_DIR / "animedl2m_real" / "images"
ANIMEDL2M_FAKE = RAW_DIR / "animedl2m_fake" / "images"

DANBOORU2021 = RAW_DIR / "danbooru2021_SQLite" / "images"
PIXIV2020 = RAW_DIR / "Pixiv_Popular" / "images"

NOVELAI3 = RAW_DIR / "novelai3" / "images"


# ============================================================
# FUNGSI PENGAMBILAN GAMBAR
# ============================================================
def get_images(folder):
    if not folder.exists():
        print(f"[WARNING] Folder tidak ditemukan: {folder}")
        return []

    return sorted([
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ============================================================
# FUNGSI RECREATE DIRECTORY
# ============================================================
def recreate_directory(folder):
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FUNGSI MEMBUAT DIRECTORY CLASS
# ============================================================
def create_class_directory(base_dir):
    (base_dir / "human").mkdir(
        parents=True,
        exist_ok=True
    )

    (base_dir / "ai").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FUNGSI COPY IMAGE
# ============================================================
def copy_images(images, destination, prefix="image"):
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(images):
        destination_file = (
            destination /
            f"{prefix}_{index:05d}{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination_file
        )


# ============================================================
# VALIDASI STANDAR
# Hanya Dimensi Piksel & Rasio Aspek
# ============================================================
def is_valid_human_image(image_path):
    """
    Validasi Cepat Citra Human Tanpa Filter AI Berat.
    Hanya mengecek ukuran dan rasio kelonjongan agar tidak memicu bottleneck.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # ------------------------------------------------
            # 1. Cek Batas Dimensi Dasar
            # ------------------------------------------------
            if (
                width < MIN_WIDTH
                or height < MIN_HEIGHT
                or width > MAX_WIDTH
                or height > MAX_HEIGHT
            ):
                return False

            # ------------------------------------------------
            # 2. Cek Kesetaraan Rasio Aspek (<= 2.0)
            # ------------------------------------------------
            aspect_ratio = max(
                width / height,
                height / width
            )

            if aspect_ratio > MAX_ASPECT_RATIO:
                return False

            return True

    except Exception:
        return False


# ============================================================
# FILTER HUMAN IMAGE
# ============================================================
def filter_human_images(
    images,
    dataset_name="HUMAN"
):
    valid = []

    for image in images:
        if is_valid_human_image(image):
            valid.append(image)

    print(
        f"[{dataset_name}] | "
        f"Raw: {len(images)} | "
        f"Valid: {len(valid)} | "
        f"Rejected: {len(images) - len(valid)}"
    )

    return valid


# ============================================================
# PREPARE ANIMEDL-2M HUMAN
# ============================================================
def prepare_animedl2m_human(human_images):
    valid = filter_human_images(
        human_images,
        "HUMAN / ANIMEDL-2M REAL"
    )

    if len(valid) < HUMAN_ANIMEDL2M_COUNT:
        raise RuntimeError(
            f"AnimeDL-2M Real valid hanya "
            f"{len(valid)}, diperlukan "
            f"{HUMAN_ANIMEDL2M_COUNT}."
        )

    rng = random.Random(RANDOM_SEED)

    return rng.sample(
        valid,
        HUMAN_ANIMEDL2M_COUNT
    )


# ============================================================
# CLASSIFY ANIMEDL-2M FAKE
# ============================================================
def classify_animedl2m_fake(fake_images):
    groups = {
        "flux": [],
        "sdxl": [],
        "sd": []
    }

    for image in fake_images:
        name = image.name.lower()

        if name.startswith("flux"):
            groups["flux"].append(image)

        elif name.startswith("sdxl"):
            groups["sdxl"].append(image)

        elif name.startswith("sd_"):
            groups["sd"].append(image)

    return groups


# ============================================================
# PREPARE ANIMEDL-2M FAKE
# ============================================================
def prepare_animedl2m_fake(fake_images):
    groups = classify_animedl2m_fake(fake_images)

    required = {
        "flux": FLUX_COUNT,
        "sdxl": SDXL_COUNT,
        "sd": SD_COUNT
    }

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in (
        "flux",
        "sdxl",
        "sd"
    ):
        part = rng.sample(
            groups[key],
            required[key]
        )

        selected.extend(part)

    rng.shuffle(selected)

    return selected


# ============================================================
# SEPARASI JALUR EVALUASI CROSS-GENERATOR UNSEEN HUMAN
# ============================================================
def prepare_danbooru2021(danbooru_images):
    valid = filter_human_images(
        danbooru_images,
        "HUMAN / DANBOORU2021_SQLITE"
    )

    if len(valid) < DANBOORU_COUNT:
        raise RuntimeError(
            f"Danbooru2021 valid hanya "
            f"{len(valid)}, diperlukan "
            f"{DANBOORU_COUNT}."
        )

    rng = random.Random(
        RANDOM_SEED + 2
    )

    return rng.sample(
        valid,
        DANBOORU_COUNT
    )


def prepare_pixiv2020(pixiv_images):
    valid = filter_human_images(
        pixiv_images,
        "HUMAN / PIXIV 2020"
    )

    if len(valid) < PIXIV_COUNT:
        raise RuntimeError(
            f"Pixiv 2020 valid hanya "
            f"{len(valid)}, diperlukan "
            f"{PIXIV_COUNT}."
        )

    rng = random.Random(
        RANDOM_SEED + 2
    )

    return rng.sample(
        valid,
        PIXIV_COUNT
    )


# ============================================================
# CLASSIFY NOVELAI 3
# ============================================================
def classify_novelai3(novelai_images):
    groups = {
        "832x1216": [],
        "1024x1024": [],
        "1216x832": []
    }

    for image in novelai_images:
        path_string = str(image).lower()

        if (
            "832x1216" in path_string
            or "832_1216" in path_string
        ):
            groups["832x1216"].append(image)

        elif (
            "1024x1024" in path_string
            or "1024_1024" in path_string
        ):
            groups["1024x1024"].append(image)

        elif (
            "1216x832" in path_string
            or "1216_832" in path_string
        ):
            groups["1216x832"].append(image)

    return groups


# ============================================================
# PREPARE NOVELAI
# ============================================================
def prepare_novelai(novelai_images):
    groups = classify_novelai3(novelai_images)

    required = {
        "832x1216": NOVELAI_832_COUNT,
        "1024x1024": NOVELAI_1024_COUNT,
        "1216x832": NOVELAI_1216_COUNT
    }

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in (
        "832x1216",
        "1024x1024",
        "1216x832"
    ):
        part = rng.sample(
            groups[key],
            required[key]
        )

        selected.extend(part)

    rng.shuffle(selected)

    return selected


# ============================================================
# INTEGRASI FUNGSI PREPARED DATASET
# DINAMIS & BEBAS BENTROK
# ============================================================
def create_prepared_dataset(
    human_animedl2m,
    animedl2m_fake,
    human_cross,
    novelai,
    prefix_human="danbooru2021"
):
    recreate_directory(PREPARED_DIR)

    copy_images(
        human_animedl2m,
        PREPARED_DIR /
        "human_animedl2m_real" /
        "images",
        prefix="human_animedl2m"
    )

    copy_images(
        animedl2m_fake,
        PREPARED_DIR /
        "ai_train_animedl2m" /
        "images",
        prefix="animedl2m"
    )

    copy_images(
        novelai,
        PREPARED_DIR /
        "ai_cross_novelai3" /
        "images",
        prefix="novelai3"
    )

    folder_name = (
        "human_danbooru2021"
        if prefix_human == "danbooru2021"
        else "human_pixiv2020"
    )

    copy_images(
        human_cross,
        PREPARED_DIR /
        folder_name /
        "images",
        prefix=prefix_human
    )


# ============================================================
# SPLIT EXACT
# ============================================================
def split_exact(
    images,
    train_count,
    val_count,
    test_count,
    seed
):
    total = (
        train_count +
        val_count +
        test_count
    )

    if len(images) != total:
        raise ValueError(
            f"Jumlah gambar {len(images)} "
            f"tidak sama dengan total split {total}."
        )

    rng = random.Random(seed)

    items = list(images)

    rng.shuffle(items)

    return (
        items[:train_count],
        items[
            train_count:
            train_count + val_count
        ],
        items[
            train_count + val_count:
        ]
    )


# ============================================================
# CREATE IN-DOMAIN SPLIT
# ============================================================
def create_in_domain_split():
    human = get_images(
        PREPARED_DIR /
        "human_animedl2m_real" /
        "images"
    )

    ai = get_images(
        PREPARED_DIR /
        "ai_train_animedl2m" /
        "images"
    )

    human_train, human_val, human_test = split_exact(
        human,
        TRAIN_PER_CLASS,
        VAL_PER_CLASS,
        TEST_IN_DOMAIN_PER_CLASS,
        RANDOM_SEED
    )

    ai_train, ai_val, ai_test = split_exact(
        ai,
        TRAIN_PER_CLASS,
        VAL_PER_CLASS,
        TEST_IN_DOMAIN_PER_CLASS,
        RANDOM_SEED + 1
    )

    for name in (
        "train",
        "validation",
        "test_in_domain"
    ):
        recreate_directory(
            SPLIT_DIR / name
        )

        create_class_directory(
            SPLIT_DIR / name
        )

    copy_images(
        human_train,
        SPLIT_DIR / "train" / "human",
        "human"
    )

    copy_images(
        ai_train,
        SPLIT_DIR / "train" / "ai",
        "ai"
    )

    copy_images(
        human_val,
        SPLIT_DIR / "validation" / "human",
        "human"
    )

    copy_images(
        ai_val,
        SPLIT_DIR / "validation" / "ai",
        "ai"
    )

    copy_images(
        human_test,
        SPLIT_DIR / "test_in_domain" / "human",
        "human"
    )

    copy_images(
        ai_test,
        SPLIT_DIR / "test_in_domain" / "ai",
        "ai"
    )


# ============================================================
# CREATE CROSS-GENERATOR TEST
# ============================================================
def create_cross_generator_test(
    prefix_human="danbooru2021"
):
    folder_name = (
        "human_danbooru2021"
        if prefix_human == "danbooru2021"
        else "human_pixiv2020"
    )

    human_unseen = get_images(
        PREPARED_DIR /
        folder_name /
        "images"
    )

    novelai = get_images(
        PREPARED_DIR /
        "ai_cross_novelai3" /
        "images"
    )

    cross_dir = (
        SPLIT_DIR /
        "test_cross_generator"
    )

    recreate_directory(cross_dir)

    create_class_directory(cross_dir)

    copy_images(
        human_unseen,
        cross_dir / "human",
        "human"
    )

    copy_images(
        novelai,
        cross_dir / "ai",
        "ai"
    )


# ============================================================
# UTAMA (MAIN JALUR EKSEKUSI)
# ============================================================
def main():

    # --------------------------------------------------------
    # SAKELAR MANUAL UTAMA 
    # --------------------------------------------------------
    MODE = "PIXIV"
    # MODE = "DANBOORU"

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------
    human_animedl2m = get_images(
        ANIMEDL2M_REAL
    )

    fake = get_images(
        ANIMEDL2M_FAKE
    )

    novelai = get_images(
        NOVELAI3
    )

    # --------------------------------------------------------
    # PREPARE DATASET
    # --------------------------------------------------------
    selected_human_animedl2m = (
        prepare_animedl2m_human(
            human_animedl2m
        )
    )

    selected_fake = (
        prepare_animedl2m_fake(
            fake
        )
    )

    selected_novelai = (
        prepare_novelai(
            novelai
        )
    )

    # --------------------------------------------------------
    # MODE DANBOORU
    # --------------------------------------------------------
    if MODE == "DANBOORU":

        raw_danbooru = get_images(
            DANBOORU2021
        )

        selected_human_cross = (
            prepare_danbooru2021(
                raw_danbooru
            )
        )

        create_prepared_dataset(
            selected_human_animedl2m,
            selected_fake,
            selected_human_cross,
            selected_novelai,
            prefix_human="danbooru2021"
        )

        recreate_directory(
            SPLIT_DIR
        )

        create_in_domain_split()

        create_cross_generator_test(
            prefix_human="danbooru2021"
        )

    # --------------------------------------------------------
    # MODE PIXIV
    # --------------------------------------------------------
    elif MODE == "PIXIV":

        raw_pixiv = get_images(
            PIXIV2020
        )

        selected_human_cross = (
            prepare_pixiv2020(
                raw_pixiv
            )
        )

        create_prepared_dataset(
            selected_human_animedl2m,
            selected_fake,
            selected_human_cross,
            selected_novelai,
            prefix_human="pixiv2020"
        )

        recreate_directory(
            SPLIT_DIR
        )

        create_in_domain_split()

        create_cross_generator_test(
            prefix_human="pixiv2020"
        )

    print(
        f"\n[SUKSES] Dataset Preparation "
        f"Berhasil Selesai dalam Mode: {MODE}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()