from pathlib import Path
import shutil
import tarfile
import zipfile
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("raw")
DOWNLOAD_DIR = Path("downloads")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# DEPENDENCY
# ============================================================

def install_package(package):
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        package,
    ])


def check_dependencies():

    try:
        import huggingface_hub
    except ImportError:
        print("[INFO] Installing huggingface_hub...")
        install_package("huggingface_hub")

    try:
        import kagglehub
    except ImportError:
        print("[INFO] Installing kagglehub...")
        install_package("kagglehub")


# ============================================================
# DIRECTORY
# ============================================================

def prepare_image_dir(dataset_name):

    image_dir = RAW_DIR / dataset_name / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return image_dir


# ============================================================
# H.1 AnimeDL-2M
# FAKE / AI
# ============================================================

def download_animedl2m_fake():

    from huggingface_hub import hf_hub_download

    print("\n" + "=" * 70)
    print("H.1 AnimeDL-2M — FAKE / AI")
    print("=" * 70)

    output_dir = prepare_image_dir(
        "animedl2m_fake"
    )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading AnimeDL-2M fake archive...")

    archive_path = hf_hub_download(
        repo_id="FlyTweety/AnimeDL-2M",
        filename="fake_images/0000.tar.gz",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )

    archive_path = Path(archive_path)

    print(f"Archive: {archive_path}")

    print("\nExtracting images...")

    extracted = 0

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as tar:

        for member in tar.getmembers():

            if not member.isfile():
                continue

            filename = Path(
                member.name
            ).name

            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = output_dir / filename

            # Avoid filename collision
            if destination.exists():

                stem = destination.stem
                suffix = destination.suffix

                counter = 1

                while destination.exists():

                    destination = (
                        output_dir
                        / f"{stem}_{counter}{suffix}"
                    )

                    counter += 1

            source = tar.extractfile(member)

            if source is None:
                continue

            with open(destination, "wb") as f:
                shutil.copyfileobj(
                    source,
                    f,
                )

            extracted += 1

    print(f"\n[SUCCESS] Extracted {extracted:,} images.")

    print(
        f"Saved to: {output_dir}"
    )


# ============================================================
# H.2 Danbooru2024-SFW
# REAL / HUMAN
# ============================================================

def download_danbooru_real():

    from huggingface_hub import snapshot_download

    print("\n" + "=" * 70)
    print("H.2 Danbooru2024-SFW — REAL / HUMAN")
    print("=" * 70)

    dataset_dir = (
        RAW_DIR
        / "animedl2m_real"
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading Danbooru images...")

    print(
        "Repository: deepghs/danbooru2024-sfw"
    )

    snapshot_download(
        repo_id="deepghs/danbooru2024-sfw",
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns=[
            "images/*",
        ],
    )

    image_dir = dataset_dir / "images"

    if not image_dir.exists():

        raise RuntimeError(
            "Folder images/ tidak ditemukan "
            "setelah download Danbooru."
        )

    image_count = sum(
        1
        for file in image_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    print(
        f"\n[SUCCESS] Found {image_count:,} images."
    )

    print(
        f"Saved to: {image_dir}"
    )


# ============================================================
# H.3 NovelAI3
# UNSEEN AI
# ============================================================

def download_novelai3():

    from huggingface_hub import hf_hub_download

    print("\n" + "=" * 70)
    print("H.3 NovelAI3 — UNSEEN AI")
    print("=" * 70)

    output_dir = prepare_image_dir(
        "novelai3"
    )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading NovelAI3 archive...")

    archive_path = hf_hub_download(
        repo_id="shareAI/novelai3",
        filename="none.zip",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )

    archive_path = Path(archive_path)

    print(
        f"Archive: {archive_path}"
    )

    print("\nExtracting images...")

    extracted = 0

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as z:

        for member in z.infolist():

            if member.is_dir():
                continue

            filename = Path(
                member.filename
            ).name

            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = output_dir / filename

            # Avoid filename collision
            if destination.exists():

                stem = destination.stem
                suffix = destination.suffix

                counter = 1

                while destination.exists():

                    destination = (
                        output_dir
                        / f"{stem}_{counter}{suffix}"
                    )

                    counter += 1

            with z.open(member) as source:
                with open(destination, "wb") as target:

                    shutil.copyfileobj(
                        source,
                        target,
                    )

            extracted += 1

    print(
        f"\n[SUCCESS] Extracted {extracted:,} images."
    )

    print(
        f"Saved to: {output_dir}"
    )


# ============================================================
# H.4 Pixiv Popular Illustrations
# UNSEEN HUMAN
# ============================================================

def download_pixiv():

    import kagglehub

    print("\n" + "=" * 70)
    print("H.4 Pixiv Popular Illustrations — UNSEEN HUMAN")
    print("=" * 70)

    output_dir = prepare_image_dir(
        "Pixiv_Popular"
    )

    print(
        "\nDownloading Kaggle dataset..."
    )

    downloaded_path = kagglehub.dataset_download(
        "profnote/pixiv-popular-illustrations"
    )

    downloaded_path = Path(
        downloaded_path
    )

    print(
        f"Downloaded to: {downloaded_path}"
    )

    print("\nCopying images...")

    copied = 0

    for file in downloaded_path.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        destination = output_dir / file.name

        # Avoid filename collision
        if destination.exists():

            stem = destination.stem
            suffix = destination.suffix

            counter = 1

            while destination.exists():

                destination = (
                    output_dir
                    / f"{stem}_{counter}{suffix}"
                )

                counter += 1

        shutil.copy2(
            file,
            destination,
        )

        copied += 1

    print(
        f"\n[SUCCESS] Copied {copied:,} images."
    )

    print(
        f"Saved to: {output_dir}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def count_images():

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    datasets = [
        "animedl2m_fake",
        "animedl2m_real",
        "novelai3",
        "Pixiv_Popular",
    ]

    for dataset in datasets:

        image_dir = (
            RAW_DIR
            / dataset
            / "images"
        )

        if not image_dir.exists():

            print(
                f"{dataset:<25} NOT FOUND"
            )

            continue

        count = sum(
            1
            for file in image_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        print(
            f"{dataset:<25} {count:,} images"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANIME AI vs HUMAN-MADE DATASET DOWNLOADER")
    print("=" * 70)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_dependencies()

    # --------------------------------------------------------
    # H.1
    # AnimeDL-2M = FAKE / AI
    # --------------------------------------------------------

    try:

        download_animedl2m_fake()

    except Exception as e:

        print(
            "\n[ERROR] AnimeDL-2M:"
        )

        print(e)

    # --------------------------------------------------------
    # H.2
    # Danbooru = REAL / HUMAN
    # --------------------------------------------------------

    try:

        download_danbooru_real()

    except Exception as e:

        print(
            "\n[ERROR] Danbooru:"
        )

        print(e)

    # --------------------------------------------------------
    # H.3
    # NovelAI3 = UNSEEN AI
    # --------------------------------------------------------

    try:

        download_novelai3()

    except Exception as e:

        print(
            "\n[ERROR] NovelAI3:"
        )

        print(e)

    # --------------------------------------------------------
    # H.4
    # Pixiv = UNSEEN HUMAN
    # --------------------------------------------------------

    try:

        download_pixiv()

    except Exception as e:

        print(
            "\n[ERROR] Pixiv:"
        )

        print(e)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    count_images()

    print("\n" + "=" * 70)
    print("DOWNLOAD SELESAI")
    print("=" * 70)

    print(
        f"\nDataset berada di:"
        f"\n{RAW_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
