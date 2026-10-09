#!/usr/bin/env bash
# 为 Ubuntu 18.04 构建发布脚本所需的 patchelf 版本。
set -Eeuo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
prefix="${repo_dir}/3rdparty/tools"
cache="${prefix}/build_cache"
version=0.14.3
sha256=8fabf4210499744ced101612cd5c9fd12b94af67a16297cb5d3ff682c007ffdb
archive="${cache}/patchelf-${version}.tar.gz"
jobs="${JOBS:-2}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
if [[ -x "${prefix}/bin/patchelf" && "$("${prefix}/bin/patchelf" --version)" == "patchelf ${version}" ]]; then
    echo "patchelf ${version} already installed at ${prefix}"
    exit 0
fi
for tool in wget sha256sum tar make g++; do
    command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done
mkdir -p "$cache/tmp"
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
trap 'echo "patchelf build failed at line ${LINENO}; cache retained at ${cache}" >&2' ERR
if [[ ! -f "$archive" ]]; then
    wget --timeout=60 --tries=3 -c -O "$archive.part" \
        "https://github.com/NixOS/patchelf/releases/download/${version}/patchelf-${version}.tar.gz"
    mv -- "$archive.part" "$archive"
fi
echo "$sha256  $archive" | sha256sum -c -
if [[ ! -f "$cache/patchelf-$version/configure" ]]; then
    tar -xzf "$archive" -C "$cache"
fi
cd "$cache/patchelf-$version"
./configure --prefix="$prefix"
make -j"$jobs"
make install
"$prefix/bin/patchelf" --version
