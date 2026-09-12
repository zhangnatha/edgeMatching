#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
qt_version="5.15.16"
qt_url="https://download.qt.io/archive/qt/5.15/${qt_version}/single/qt-everywhere-opensource-src-${qt_version}.tar.xz"
qt_sha256_default="efa99827027782974356aceff8a52bd3d2a8a93a54dd0db4cca41b5e35f1041c"
qt_sha256="${QT_SHA256:-${qt_sha256_default}}"

# 1. 默认缓存目录修改为脚本同级目录下的 build_cache 文件夹
cache_dir="${QT_CACHE_DIR:-${script_dir}/build_cache}"
install_dir="${QT_INSTALL_DIR:-${repo_dir}/3rdparty/qt5}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

usage() {
    echo "Usage: $0 [--jobs N] [--cache DIR] [--install DIR]"
    echo "Environment overrides: JOBS, QT_SHA256, QT_CACHE_DIR, QT_SRC_DIR, QT_BUILD_DIR, QT_ARCHIVE, QT_INSTALL_DIR"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; jobs="$2"; shift 2 ;;
        --cache) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; cache_dir="$2"; shift 2 ;;
        --install) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; install_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${QT_SRC_DIR:-}" ]]; then
    src_dir="${cache_dir}/qt-everywhere-opensource-src-${qt_version}"
else
    src_dir="${QT_SRC_DIR}"
fi
if [[ -z "${QT_BUILD_DIR:-}" ]]; then
    build_dir="${cache_dir}/build-${qt_version}"
else
    build_dir="${QT_BUILD_DIR}"
fi
if [[ -z "${QT_ARCHIVE:-}" ]]; then
    archive="${cache_dir}/qt-everywhere-opensource-src-${qt_version}.tar.xz"
else
    archive="${QT_ARCHIVE}"
fi

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
for command in tar sha256sum make awk perl python3 gcc g++; do
    command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done
if command -v curl >/dev/null; then
    downloader=(curl -fL --retry 3 --retry-delay 2 -o)
elif command -v wget >/dev/null; then
    downloader=(wget -O)
else
    echo "Missing downloader: install curl or wget" >&2
    exit 1
fi

if [[ -x "${install_dir}/bin/qmake" ]]; then
    echo "Qt ${qt_version} already installed at ${install_dir}; nothing to do."
    exit 0
fi

# 2. 创建清理函数，注册 trap 捕捉脚本退出（无论是正常完成还是报错中断）
cleanup() {
    if [[ -d "${cache_dir}" ]]; then
        echo "Cleaning up temporary build files in ${cache_dir}..."
        rm -rf "${cache_dir}"
    fi
}
trap cleanup EXIT

mkdir -p "${cache_dir}"
if [[ ! -f "${archive}" ]]; then
    echo "Downloading Qt ${qt_version} to ${cache_dir}..."
    "${downloader[@]}" "${archive}" "${qt_url}"
fi
actual_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${qt_sha256}" ]]; then
    echo "Qt archive SHA256 mismatch: expected ${qt_sha256}, got ${actual_sha256}" >&2
    exit 1
fi

# 3. 使用 --strip-components=1 规范解压目录，避免之前的 configure 文件未找到问题
if [[ ! -f "${src_dir}/configure" ]]; then
    echo "Extracting Qt source..."
    mkdir -p "${src_dir}"
    tar -xJf "${archive}" -C "${src_dir}" --strip-components=1
fi

mkdir -p "${build_dir}"
pushd "${build_dir}" >/dev/null
if [[ ! -f Makefile ]]; then
    "${src_dir}/configure" \
        -prefix "${install_dir}" \
        -opensource -confirm-license \
        -nomake examples -nomake tests \
        -skip qt3d -skip qtactiveqt -skip qtandroidextras -skip qtcharts \
        -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative \
        -skip qtgamepad -skip qtlocation -skip qtlottie -skip qtmultimedia \
        -skip qtnetworkauth -skip qtpurchasing -skip qtquick3d -skip qtremoteobjects \
        -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus \
        -skip qtserialport -skip qtspeech -skip qttools -skip qttranslations \
        -skip qtvirtualkeyboard -skip qtwebchannel -skip qtwebglplugin \
        -skip qtwebsockets -skip qtwebview -skip qtwinextras -skip qtx11extras
fi
make -j"${jobs}"
make install
popd >/dev/null

echo "Qt ${qt_version} installed at ${install_dir}."
