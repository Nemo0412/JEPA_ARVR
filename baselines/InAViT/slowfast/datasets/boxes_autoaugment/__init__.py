"""
Source:
https://github.com/Paperspace/DataAugmentationForObjectDetection
"""
try:
    from .autoaugment import rand_augment_transform
except Exception:  # imgaug is incompatible with NumPy 2
    rand_augment_transform = None