#!/usr/bin/env bash
# Download imagenette2-320 (smaller variant, 320px shortest side).
set -euo pipefail

DEST="${1:-./data}"
mkdir -p "$DEST"
cd "$DEST"

URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
echo "downloading from $URL ..."
curl -L "$URL" -o imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
rm -f imagenette2-320.tgz
echo "done. data at $(pwd)/imagenette2-320"
