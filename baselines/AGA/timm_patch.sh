#!/bin/sh
pip install timm==v0.5.4
PACKAGELOC=`pip show timm | grep Location | cut -d ' ' -f 2`
cp patch_swin_transformer.py $PACKAGELOC/timm/models/swin_transformer.py
