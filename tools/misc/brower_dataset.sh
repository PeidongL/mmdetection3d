#!/usr/bin/env bash

python tools/misc/browse_dataset.py configs/L3_data_models/bevfusion/bevdepth_fusion.py \
--task plus-det \
 --online  \
 --output-dir \
 vis, --aug