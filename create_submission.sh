#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./create_submission.sh <TEAM_ID>"
    echo "Example: ./create_submission.sh 1234"
    exit 1
fi

TEAM_ID=$1
MASK_SRC="task_1a_binary.png"
SCRIPT_SRC="src/task_1a/task_1a/camera_detection.py"
ZIP_NAME="HE_${TEAM_ID}_task_1a.zip"

if [ ! -f "$MASK_SRC" ]; then
    echo "Error: $MASK_SRC not found. Make sure camera_detection has executed once."
    exit 1
fi

cp "$MASK_SRC" "HE_${TEAM_ID}_binary.png"
cp "$SCRIPT_SRC" "HE_${TEAM_ID}_camera_detection.py"

zip -j "$ZIP_NAME" "HE_${TEAM_ID}_binary.png" "HE_${TEAM_ID}_camera_detection.py"

rm -f "HE_${TEAM_ID}_binary.png" "HE_${TEAM_ID}_camera_detection.py"

echo "Created submission zip: $ZIP_NAME"
unzip -l "$ZIP_NAME"
