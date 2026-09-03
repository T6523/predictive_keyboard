#!/usr/bin/env bash
# Builds KenLM's lmplz (trainer) and build_binary (arpa -> binary) into scripts/bin/.
# One-time setup. Needs: cmake libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
#   sudo apt-get install -y cmake libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --depth 1 https://github.com/kpu/kenlm.git "$work/kenlm"
cmake -S "$work/kenlm" -B "$work/kenlm/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$work/kenlm/build" -j"$(nproc)" --target lmplz build_binary

mkdir -p "$here/bin"
cp "$work/kenlm/build/bin/lmplz" "$work/kenlm/build/bin/build_binary" "$here/bin/"
echo "built -> $here/bin/{lmplz,build_binary}"
