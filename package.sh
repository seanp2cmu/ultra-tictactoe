#!/bin/bash

echo "================================================"
echo "Ultra Tic-Tac-Toe Packaging Script"
echo "================================================"

PROJECT_NAME="ultra-tictacto"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${PROJECT_NAME}_${TIMESTAMP}.tar.gz"

echo "1. Generating requirements.txt..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    pip freeze > requirements.txt
    deactivate
    echo "   ✓ requirements.txt created"
else
    echo "   ⚠ .venv not found, skipping requirements.txt generation"
fi

echo ""
echo "2. Creating temporary directory..."
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${PROJECT_NAME}"
mkdir -p "${PACKAGE_DIR}"

echo "   ✓ Temporary directory: ${TEMP_DIR}"

echo ""
echo "3. Copying files (excluding .venv, __pycache__, etc.)..."

rsync -av \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='*.pth' \
    --exclude='model/' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    --exclude='dist/' \
    --exclude='build/' \
    ./ "${PACKAGE_DIR}/"

echo "   ✓ Files copied"

echo ""
echo "4. Creating archive..."
cd "${TEMP_DIR}"
tar -czf "${OUTPUT_FILE}" "${PROJECT_NAME}"
mv "${OUTPUT_FILE}" "${OLDPWD}/"
cd "${OLDPWD}"

echo "   ✓ Archive created: ${OUTPUT_FILE}"

echo ""
echo "5. Cleaning up..."
rm -rf "${TEMP_DIR}"
echo "   ✓ Temporary files removed"

echo ""
echo "================================================"
echo "Package complete!"
echo "================================================"
echo "Output file: ${OUTPUT_FILE}"
echo "Size: $(du -h ${OUTPUT_FILE} | cut -f1)"
echo ""
echo "To extract:"
echo "  tar -xzf ${OUTPUT_FILE}"
echo "================================================"
