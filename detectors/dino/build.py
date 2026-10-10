"""
DINO builder (DINO-4scale, ResNet-50, 900 queries, contrastive denoising, mixed query selection), Image task.

Paper: Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection" (ICLR 2023,
       arXiv 2203.03605).

Code : official IDEA-Research/DINO (models/dino, config DINO_4scale.py) - model files copied unchanged into detectors/dino/official/ except the imports
       (util.* -> detectors/detr/official misc_min + the repository's own box_ops; backbone -> detectors/detr/adapter.py)
       and the [Framework] fixes marked in the files. Built by the repository's own build function with its own default
       arguments (official/args_official.py) through detectors/detr/family.py.
       Deformable attention: the PyTorch reference ms_deform_attn_core_pytorch of the repository (no CUDA op).

[Unified] no ImageNet weights, trainable BN + PreciseBN, common init (nn.Conv2d / BN only), no mean / std, fixed 608
input without padding masks, Adam 1e-4 for all parameters (official AdamW, backbone 1e-5) with the official gradient
clipping 0.1, 30 epochs, common augmentation and post-processing (NMS on top of the NMS-free set prediction).
"""

from detectors.detr.family import OfficialDETRFamily
from detectors.dino.official.dino import build_dino
from detectors.dino.official.args_official import ARGS


def build_model(task, num_classes):
    return OfficialDETRFamily(num_classes, build_dino, ARGS, call="targets", overrides=None)
