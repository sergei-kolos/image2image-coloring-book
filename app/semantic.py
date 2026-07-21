from __future__ import annotations

import numpy as np
import cv2

_MODEL = None
_PROCESSOR = None


def is_semantic_available() -> bool:
    """True if transformers + SegFormer model can be loaded."""
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation
        _ = torch.nn.Module
        _ = SegformerForSemanticSegmentation
        return True
    except ImportError:
        return False


def _load_semantic_model():
    """Lazy-load SegFormer-B2 on ADE20K (cached globally)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from .config import SEMANTIC_MODEL_NAME, AI_DEVICE

    _PROCESSOR = SegformerImageProcessor.from_pretrained(SEMANTIC_MODEL_NAME)
    _MODEL = SegformerForSemanticSegmentation.from_pretrained(SEMANTIC_MODEL_NAME)
    _MODEL.to(AI_DEVICE)
    _MODEL.eval()


def compute_importance_map(image: np.ndarray) -> np.ndarray:
    """Return per-pixel importance map (H x W, float32, 0..1).

    1.0 = preserve detail (faces, hands, text), 0.0 = simplify (sky, walls).
    """
    from .config import AI_DEVICE
    import torch

    h, w = image.shape[:2]

    _load_semantic_model()

    inputs = _PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(AI_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL(**inputs)
        logits = outputs.logits  # (1, 150, H/4, W/4)

    seg_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    importance_small = _CLASS_IMPORTANCE[seg_map]

    importance = cv2.resize(importance_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    importance = cv2.GaussianBlur(importance, (31, 31), 8)
    return np.clip(importance, 0.0, 1.0)


# ADE20K 150-class importance mapping
# Index = ADE20K class id, value = importance (0..1)
_CLASS_IMPORTANCE = np.array([
    0.05,  # 0: wall
    0.15,  # 1: building
    0.05,  # 2: sky
    0.10,  # 3: floor
    0.25,  # 4: tree
    0.10,  # 5: ceiling
    0.15,  # 6: road
    0.35,  # 7: bed
    0.20,  # 8: windowpane
    0.05,  # 9: grass
    0.30,  # 10: cabinet
    0.10,  # 11: sidewalk
    1.00,  # 12: person
    0.05,  # 13: earth/ground
    0.15,  # 14: door
    0.35,  # 15: table
    0.20,  # 16: mountain
    0.30,  # 17: plant
    0.20,  # 18: curtain
    0.40,  # 19: chair
    0.50,  # 20: car
    0.05,  # 21: water
    0.50,  # 22: painting
    0.35,  # 23: sofa
    0.25,  # 24: shelf
    0.20,  # 25: house
    0.05,  # 26: sea
    0.40,  # 27: mirror
    0.25,  # 28: rug
    0.05,  # 29: field
    0.35,  # 30: armchair
    0.30,  # 31: seat
    0.20,  # 32: fence
    0.35,  # 33: desk
    0.15,  # 34: rock/stone
    0.30,  # 35: wardrobe
    0.35,  # 36: lamp
    0.30,  # 37: bathtub
    0.20,  # 38: railing
    0.25,  # 39: cushion
    0.15,  # 40: base/pedestal
    0.30,  # 41: box
    0.15,  # 42: column
    0.60,  # 43: signboard
    0.30,  # 44: chest of drawers
    0.25,  # 45: counter
    0.05,  # 46: sand
    0.30,  # 47: sink
    0.15,  # 48: skyscraper
    0.30,  # 49: fireplace
    0.25,  # 50: refrigerator
    0.20,  # 51: grandstand
    0.15,  # 52: path
    0.15,  # 53: stairs
    0.15,  # 54: runway
    0.30,  # 55: case/display case
    0.30,  # 56: pool table
    0.25,  # 57: pillow
    0.20,  # 58: screen door
    0.15,  # 59: stairway
    0.10,  # 60: river
    0.25,  # 61: bridge
    0.25,  # 62: bookcase
    0.20,  # 63: blind/screen
    0.30,  # 64: coffee table
    0.30,  # 65: toilet
    0.70,  # 66: flower
    0.30,  # 67: book
    0.15,  # 68: hill
    0.30,  # 69: bench
    0.25,  # 70: countertop
    0.30,  # 71: stove
    0.25,  # 72: palm
    0.30,  # 73: kitchen
    0.60,  # 74: computer
    0.40,  # 75: swivel chair
    0.50,  # 76: boat
    0.30,  # 77: bar
    0.40,  # 78: arcade machine
    0.10,  # 79: hovel/hut
    0.50,  # 80: bus
    0.30,  # 81: towel
    0.35,  # 82: light
    0.50,  # 83: truck
    0.15,  # 84: tower
    0.40,  # 85: chandelier
    0.20,  # 86: awning
    0.25,  # 87: streetlight
    0.30,  # 88: booth
    0.55,  # 89: television
    0.50,  # 90: airplane
    0.10,  # 91: dirt track
    0.60,  # 92: apparel/clothing
    0.15,  # 93: pole
    0.10,  # 94: land/ground
    0.20,  # 95: bannister
    0.15,  # 96: escalator
    0.30,  # 97: ottoman
    0.40,  # 98: bottle
    0.30,  # 99: buffet/counter
    0.40,  # 100: poster
    0.30,  # 101: stage
    0.50,  # 102: van
    0.50,  # 103: ship
    0.40,  # 104: fountain
    0.15,  # 105: conveyer belt
    0.20,  # 106: canopy
    0.25,  # 107: washer
    0.40,  # 108: plaything/toy
    0.25,  # 109: swimming pool
    0.30,  # 110: stool
    0.30,  # 111: barrel
    0.30,  # 112: basket
    0.40,  # 113: waterfall
    0.20,  # 114: tent
    0.50,  # 115: bag
    0.50,  # 116: minibike
    0.30,  # 117: cradle
    0.30,  # 118: oven
    0.40,  # 119: ball
    0.60,  # 120: food
    0.15,  # 121: step/stair
    0.30,  # 122: tank
    0.30,  # 123: trade name/brand
    0.25,  # 124: microwave
    0.40,  # 125: pot/flowerpot
    0.85,  # 126: animal
    0.50,  # 127: bicycle
    0.10,  # 128: lake
    0.25,  # 129: dishwasher
    0.30,  # 130: screen/CRT
    0.30,  # 131: blanket
    0.85,  # 132: sculptor
    0.25,  # 133: hood
    0.30,  # 134: sconce
    0.40,  # 135: vase
    0.35,  # 136: traffic light
    0.30,  # 137: tray
    0.10,  # 138: ashcan/trash can
    0.20,  # 139: island
    0.35,  # 140: flag
    0.90,  # 141: audience
    0.15,  # 142: park
    0.30,  # 143: glass
    0.45,  # 144: painting
    0.50,  # 145: clothing
    0.45,  # 146: picture
    0.90,  # 147: people
    0.10,  # 148: country
    0.25,  # 149: unknown
], dtype=np.float32)

assert len(_CLASS_IMPORTANCE) == 150, f"Expected 150 ADE20K classes, got {len(_CLASS_IMPORTANCE)}"
assert _CLASS_IMPORTANCE.min() >= 0.0 and _CLASS_IMPORTANCE.max() <= 1.0
