#!/bin/bash

BASE_DIR="/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/tools/dataset_convert"
TARGET_DIR="/media/lams/D/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_3"

TARGET_FOLDER_NAMES=("2-4" "2-5" "2-6" "4-1" "4-2" "4-3" "4-4" "4-5" "4-6" "6-1" "6-2" "6-3" "6-4" "6-5" "6-6" "N-1" "N-2")

for TARGET in "${TARGET_FOLDER_NAMES[@]}"
do
    echo "Processing target: $TARGET"
    python3 create_ATD_dataset.py $TARGET_DIR $TARGET
    echo "Completed processing target: $TARGET"
done
