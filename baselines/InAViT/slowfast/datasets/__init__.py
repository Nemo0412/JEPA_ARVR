#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa

# Dataset modules pull EK / numpy-1.x extras; skip any that fail so MotionFormer
# can still be constructed for HD-EPIC streaming (we do not use these loaders).
for _mod in (
    "ava_dataset",
    "charades",
    "imagenet",
    "kinetics",
    "ssv2",
    "epickitchens",
    "egtea",
    "epickitchens_demo",
):
    try:
        __import__(f"slowfast.datasets.{_mod}", fromlist=["*"])
    except Exception:
        pass
