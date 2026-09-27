from pathlib import Path
import tarfile
import zipfile
import shutil
import subprocess
import sys


RAW_DIR = Path("raw")


def install(package):
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        package
    ])


def setup_dependencies():
    try:
        import huggingface_hub
    except ImportError:
        install("huggingface_hub")

    try:
        import kagglehub
    except ImportError:
        install("kagglehub")


def prepare_dir(name):
    path = RAW_DIR / name / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_tar_images(tar_path, output_dir):
    """
    Extract images from .tar.gz into output_dir.
    """
    print(f"Extracting: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        for member in members:
            if not member.isfile():
                continue

            filename = Path(member.name).name

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                member.name = filename
                tar.extract(member, output_dir)

    print(f"Done -> {output_dir}")


def extract_zip_images(zip_path, output_dir):
    """
    Extract images from ZIP into output_dir.
    """
    print(f"Extracting: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():

            if member.is_dir():
                continue

            filename = Path(member.filename).name

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                target = output_dir / filename

                with z.open(member) as source:
                    with open(target, "wb") as destination:
                        shutil.copyfileobj(source, destination)

    print(f"Done -> {output_dir}")


def download_animedl2m_fake():
    """
    AnimeDL-2M fake images
    """

    from huggingface_hub import hf_hub_download

    output_dir = prepare_dir("animedl2m_fake")

    archive_dir = Path("downloads")
    archive_dir.mkdir(exist_ok=True)

    print("\n=== AnimeDL-2M FAKE ===")

    archive = hf_hub_download(
        repo_id="FlyTweety/AnimeDL-2M",
        filename="fake_images/0000.tar.gz",
        repo_type="dataset",
        local_dir=archive_dir
    )

    extract_tar_images(
        Path(archive),
        output_dir
    )


def download_novelai3():
    """
    NovelAI3
    """

    from huggingface_hub import hf_hub_download

    output_dir = prepare_dir("novelai3")

    archive_dir = Path("downloads")
    archive_dir.mkdir(exist_ok=True)

    print("\n=== NovelAI3 ===")

    archive = hf_hub_download(
        repo_id="shareAI/novelai3",
        filename="none.zip",
        repo_type="dataset",
        local_dir=archive_dir
    )

    extract_zip_images(
        Path(archive),
        output_dir
    )


def download_pixiv():
    """
    Pixiv Popular Illustrations
    """

    import kagglehub

    output_dir = prepare_dir("Pixiv_Popular")

    print("\n=== Pixiv Popular Illustrations ===")

    dataset_path = kagglehub.dataset_download(
        "profnote/pixiv-popular-illustrations"
    )

    dataset_path = Path(dataset_path)

    print(f"Downloaded to: {dataset_path}")

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    count = 0

    for file in dataset_path.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in image_extensions:
            continue

        destination = output_dir / file.name

        # Hindari overwrite jika nama file sama
        if destination.exists():
            destination = output_dir / f"{count}_{file.name}"

        shutil.copy2(
            file,
            destination
        )

        count += 1

    print(f"Copied {count} images.")
    print(f"Saved to: {output_dir}")


def main():

    RAW_DIR.mkdir(exist_ok=True)

    setup_dependencies()

    print("\n" + "=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)

    # AnimeDL-2M fake
    download_animedl2m_fake()

    # NovelAI3
    download_novelai3()

    # Pixiv
    download_pixiv()

    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)

    print("\nStruktur:")
    print("""
raw/
├── animedl2m_fake/
│   └── images/
│
├── animedl2m_real/
│   └── images/
│
├── novelai3/
│   └── images/
│
└── Pixiv_Popular/
    └── images/
""")


if __name__ == "__main__":
    main()
