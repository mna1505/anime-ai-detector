"""Factory arsitektur untuk eksperimen modified.

Model penelitian:
- MobileNetV3-Large
- ResNet50 (Gak jadi)
- EfficientNetV2-S (Gak jadi)
- DeiT-Base (Gak jadi)
- DeiT-Tiny

DeiT menggunakan package timm agar arsitektur DeiT tersedia secara eksplisit.
Install: pip install timm
"""

# ============================================================
# DOKUMENTASI COMMAND — model_factory_modified.py
# ============================================================
# FUNGSI:
#   Factory untuk membuat arsitektur model yang dipakai pipeline modified.
#
# FOKUS MODEL:
#   - MobileNetV3-Large
#   - DeiT-Tiny
#
# DEPENDENCY:
#   DeiT membutuhkan package timm.
#
# INSTALL:
#   pip install timm
#
# OPSIONAL UNTUK COMPLEXITY:
#   pip install fvcore
#   pip install thop
#
# CATATAN:
#   File ini bukan script command utama yang dijalankan langsung.
#   Ia dipanggil oleh 04_train_modified.py dan 05_evaluate_modified.py.
# ============================================================


from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

MODEL_DISPLAY_NAMES = {
    "mobilenetv3": "MobileNetV3-Large",
    # "resnet50": "ResNet50",
    # "efficientnetv2": "EfficientNetV2-S",
    # "deit_base": "DeiT-Base",
    "deit_tiny": "DeiT-Tiny",
}

MODELS = tuple(MODEL_DISPLAY_NAMES.keys())


def _require_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "Model DeiT membutuhkan package 'timm'. Install dengan: pip install timm"
        ) from exc
    return timm


def create_model(model_name: str, pretrained: bool = True, num_classes: int = 2, level: int = 1):
    """Membuat model dengan classifier biner dan penanganan setara 4 channel untuk Level 4."""
    
    # =========================================================================
    # 1. MOBILENETV3-LARGE (Pintu masuk: model.features[0][0])
    # =========================================================================
    if model_name == "mobilenetv3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        if level == 4:
            old_conv = model.features[0][0] # Alamat pintu masuk MobileNetV3
            new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3].copy_(old_conv.weight) # Kunci 3 channel RGB ImageNet
                new_conv.weight[:, 3:].copy_(old_conv.weight.mean(dim=1, keepdim=True)) # Sisip channel ke-4 DWT
            model.features[0][0] = new_conv
            
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    # # =========================================================================
    # # 2. RESNET50 (Pintu masuk: model.conv1)
    # # =========================================================================
    # elif model_name == "resnet50":
    #     weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    #     model = models.resnet50(weights=weights)
        
    #     if level == 4:
    #         old_conv = model.conv1 # Alamat pintu masuk ResNet50
    #         new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
    #                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
    #         with torch.no_grad():
    #             new_conv.weight[:, :3].copy_(old_conv.weight) # Kunci 3 channel RGB ImageNet
    #             new_conv.weight[:, 3:].copy_(old_conv.weight.mean(dim=1, keepdim=True)) # Sisip channel ke-4 DWT
    #         model.conv1 = new_conv
            
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)

    # # =========================================================================
    # # 3. EFFICIENTNETV2-S (Pintu masuk: model.features[0][0])
    # # =========================================================================
    # elif model_name == "efficientnetv2":
    #     weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    #     model = models.efficientnet_v2_s(weights=weights)
        
    #     if level == 4:
    #         old_conv = model.features[0][0] # Alamat pintu masuk EfficientNetV2 (Sama tipe seperti MobileNet)
    #         new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
    #                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
    #         with torch.no_grad():
    #             new_conv.weight[:, :3].copy_(old_conv.weight) # Kunci 3 channel RGB ImageNet
    #             new_conv.weight[:, 3:].copy_(old_conv.weight.mean(dim=1, keepdim=True)) # Sisip channel ke-4 DWT
    #         model.features[0][0] = new_conv
            
    #     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)


        # =========================================================================
    # 4. DEI-T TINY (Pintu masuk: model.patch_embed.proj) — PERBAIKAN BUG
    # =========================================================================
    elif model_name == "deit_tiny":
        timm = _require_timm()
        model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
        
        if level == 4:
            old_proj = model.patch_embed.proj # Ini adalah objek nn.Conv2d
            # Ekstrak parameter secara langsung dan aman:
            new_proj = nn.Conv2d(
                in_channels=4, 
                out_channels=old_proj.out_channels, 
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride, 
                padding=old_proj.padding
            )
            with torch.no_grad():
                new_proj.weight[:, :3].copy_(old_proj.weight) # Kunci 3 channel RGB ImageNet
                new_proj.weight[:, 3:].copy_(old_proj.weight.mean(dim=1, keepdim=True)) # Sisip channel ke-4 DWT
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
            model.patch_embed.proj = new_proj


    else:
        raise ValueError(f"Model tidak dikenal: {model_name}. Pilihan: {MODELS}")

    return model


def get_classifier_module(model, model_name: str):
    """Mengembalikan modul classification head untuk tiap arsitektur."""
    if model_name == "mobilenetv3":
        return model.classifier
    # if model_name == "resnet50":
    #     return model.fc
    # if model_name == "efficientnetv2":
    #     return model.classifier
    if model_name in {"deit_base", "deit_tiny"}:
        # timm DeiT memakai head sebagai classifier utama.
        return model.get_classifier()
    raise ValueError(f"Model tidak dikenal: {model_name}")


def split_backbone_head_parameters(model, model_name: str):
    """Membagi parameter menjadi backbone dan classification head."""
    head_module = get_classifier_module(model, model_name)
    head_ids = {id(p) for p in head_module.parameters()}

    backbone = []
    head = []
    for p in model.parameters():
        if id(p) in head_ids:
            head.append(p)
        else:
            backbone.append(p)

    return backbone, head



def calculate_complexity(model_name: str, level: int = 1):
    """Hitung FLOPs/MACs secara dinamis sesuai dengan level channel input model."""
    # Fungsi create_model sekarang ikut menerima parameter 'level'
    model = create_model(model_name, pretrained=False, level=level).cpu().eval()
    
    # Tentukan jumlah channel input dummy secara dinamis:
    # Jika level == 4 maka 4 channel (RGB + DWT), selain itu 3 channel (RGB standar)
    input_channels = 4 if level == 4 else 3
    dummy = torch.randn(1, input_channels, 224, 224)

    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, dummy)
        flops = float(analysis.total())
        del model
        return {"flops": flops, "macs": flops / 2.0, "complexity_tool": "fvcore"}
    except Exception as fvcore_error:
        fvcore_message = str(fvcore_error)

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2.0 * float(macs)
        del model
        return {"flops": flops, "macs": float(macs), "complexity_tool": "thop"}
    except Exception as thop_error:
        del model
        return {
            "flops": None,
            "macs": None,
            "complexity_tool": "unavailable",
            "complexity_error": {"fvcore": fvcore_message, "thop": str(thop_error)},
        }

