#!/usr/bin/env bash
# Downloads the meshes and textures of the living room from https://benedikt-bitterli.me/resources/
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d scenes/living-room/models ]; then
    echo "living-room: already downloaded"
    exit 0
fi
echo "living-room: downloading"
tmp=$(mktemp -d)
curl -L --fail --progress-bar -o "$tmp/living-room.zip" "https://benedikt-bitterli.me/resources/mitsuba/living-room.zip"
unzip -q "$tmp/living-room.zip" "living-room/models/*" "living-room/textures/*" -d "$tmp"
mv "$tmp/living-room/models" "$tmp/living-room/textures" scenes/living-room/
rm -r "$tmp"
