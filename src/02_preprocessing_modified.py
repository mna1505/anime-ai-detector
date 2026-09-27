"""Preprocessing dan augmentation untuk eksperimen modified.

Label ImageFolder:
    0 = ai
    1 = human

Level 1-3: augmentasi visual yang sama seperti pipeline modified sebelumnya.
Level 4: augmentasi Level 4 + transformasi Haar DWT sebagai preprocessing
         frequency-domain. SAM tidak digunakan lagi di Level 4.
"""

# ============================================================
# DOKUMENTASI COMMAND — 02_preprocessing_modified.py
# ============================================================
# Fungsi utama:
#   Menyediakan preprocessing untuk Level 1 sampai Level 4.
#
# COMMAND UTAMA:
#   Dari ROOT proyek:
#       python src/02_preprocessing_modified.py
#
#   Dari folder src:
#       python 02_preprocessing_modified.py
#
# CATATAN PENTING:
#   File ini dipakai oleh pipeline modified:
#       Level 1 = AdamW + cosine + augmentation
#       Level 2 = Level 1 + MixUp + Label Smoothing + warm-up
#       Level 3 = Level 2 + discriminative learning rate
#       Level 4 = DWT
#
#   Pemilihan level saat eksperimen dilakukan melalui
#   04_train_modified.py / 05_evaluate_modified.py.
# ============================================================


from PIL import Image, ImageOps
import random
import torch
import torch.nn.functional as F
from torchvision import transforms

MAX_SIDE = 1024
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GAUSSIAN_BLUR_P = 0.25
JPEG_COMPRESSION_P = 0.35
JPEG_QUALITY_RANGE = (55, 90)
RANDOM_ERASING_P = 0.25
COLOR_JITTER_STRENGTH = 0.20

DWT_WAVELET = "haar"
DWT_LEVEL = 1


class ResizeLongestSide:
    def __init__(self, max_side=MAX_SIDE):
        self.max_side = max_side

    def __call__(self, image):
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_side:
            return image
        scale = self.max_side / float(longest)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


class RandomJPEGCompression:
    def __init__(self, p=JPEG_COMPRESSION_P, quality_range=JPEG_QUALITY_RANGE):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image):
        if random.random() > self.p:
            return image
        from io import BytesIO
        quality = random.randint(*self.quality_range)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

class HaarDWT1Channel:
    """Mengubah tensor RGB menjadi 1-channel representasi energi high-frequency Haar DWT.
    
    Output yang dihasilkan berukuran [1, 224, 224] yang sudah dinormalisasi per gambar.
    """
    def __init__(self, output_size=MODEL_INPUT_SIZE):
        self.output_size = output_size

    @staticmethod
    def _rgb_to_gray(x):
        # x shape: [1, 3, H, W]
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    @staticmethod
    def _dwt2(x):
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) / 2.0
        lh = (x00 - x01 + x10 - x11) / 2.0
        hl = (x00 + x01 - x10 - x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return ll, lh, hl, hh

    def __call__(self, tensor):
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"HaarDWT1Channel membutuhkan tensor RGB CxHxW, diterima {tuple(tensor.shape)}")

        # 1. Ambil data asli dalam range [0, 1] sebelum dinormalisasi ImageNet
        x = tensor.float().unsqueeze(0)
        
        # 2. Konversi ke Grayscale
        gray = self._rgb_to_gray(x)
        
        # 3. Hitung DWT 1 Level
        ll, lh, hl, hh = self._dwt2(gray)

        # 4. Hitung Energi High-Frequency Magnitude
        high_frequency = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-8)

        # 5. Upsample kembali ke ukuran 224x224
        high_frequency = F.interpolate(high_frequency, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        high_frequency = high_frequency.squeeze(0) # Shape menjadi [1, 224, 224]

        # 6. Normalisasi dinamis per gambar (Z-Score) seperti di file 07
        mean = high_frequency.mean(dim=(1, 2), keepdim=True)
        std = high_frequency.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        high_frequency_normalized = (high_frequency - mean) / std

        return high_frequency_normalized


BASE_TRANSFORM = [
    ResizeLongestSide(MAX_SIDE),
    PadToSquare(fill=0),
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), antialias=True),
]


# =========================================================================
# SOLUSI PERBAIKAN Bug WINDOWS PICKLE: CALLABLE CLASS TINGKAT GLOBAL
# =========================================================================
class L4TrainFusionWindowsCompatible:
    def __init__(self, transform_rgb_base, transform_dwt_gate):
        self.transform_rgb_base = transform_rgb_base
        self.transform_dwt_gate = transform_dwt_gate

    def __call__(self, image):
        rgb_tensor = self.transform_rgb_base(image) 
        dwt_tensor = self.transform_dwt_gate(rgb_tensor) 
        rgb_normalized = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_tensor) 
        return torch.cat([rgb_normalized, dwt_tensor], dim=0)

class L4EvalFusionWindowsCompatible:
    def __init__(self, transform_rgb, transform_dwt):
        self.transform_rgb = transform_rgb
        self.transform_dwt = transform_dwt

    def __call__(self, image):
        rgb_tensor = self.transform_rgb(image)
        dwt_tensor = self.transform_dwt(rgb_tensor)
        rgb_normalized = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_tensor)
        return torch.cat([rgb_normalized, dwt_tensor], dim=0)


def get_train_transform(model_name="resnet50", level=1):
    if level not in (1, 2, 3, 4):
        raise ValueError("Level training harus 1, 2, 3, atau 4.")

    common = [
        transforms.RandomResizedCrop(MODEL_INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.90, 1.10), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=COLOR_JITTER_STRENGTH, contrast=COLOR_JITTER_STRENGTH, saturation=COLOR_JITTER_STRENGTH, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=GAUSSIAN_BLUR_P),
        RandomJPEGCompression(),
        transforms.ToTensor(),
    ]

    if level == 4:
        # Pipa fusi dibungkus ke objek kelas global agar DataLoader Windows bisa melakukan proses serialize (pickle)
        transform_rgb_base = transforms.Compose(BASE_TRANSFORM + [transforms.RandAugment(num_ops=2, magnitude=9)] + common[-2:])
        transform_dwt_gate = HaarDWT1Channel()
        return L4TrainFusionWindowsCompatible(transform_rgb_base, transform_dwt_gate)

    return transforms.Compose(
        BASE_TRANSFORM + common + [
            transforms.RandomErasing(p=RANDOM_ERASING_P, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transform(level=1):
    """Preprocessing validation/test; Level 4 memakai DWT penggabungan 4 channel."""
    if level == 4:
        # Pipa fusi dibungkus ke objek kelas global agar DataLoader Windows bisa melakukan proses serialize (pickle)
        transform_rgb = transforms.Compose(BASE_TRANSFORM + [transforms.ToTensor()])
        transform_dwt = HaarDWT1Channel()
        return L4EvalFusionWindowsCompatible(transform_rgb, transform_dwt)
        
    return transforms.Compose(BASE_TRANSFORM + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])



def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"CUDA tersedia: {name}")
        return torch.device("cuda")
    print("CUDA tidak tersedia. Menggunakan CPU.")
    return torch.device("cpu")


if __name__ == "__main__":
    sample = Image.new("RGB", (400, 600), "white")
    for level in (1, 4):
        tensor = get_eval_transform(level=level)(sample)
        print(f"Level {level}: tensor={tuple(tensor.shape)}, dtype={tensor.dtype}")
