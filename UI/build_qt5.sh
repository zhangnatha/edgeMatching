#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
qt_version="5.15.16"
qt_url="${QT_URL:-https://download.qt.io/archive/qt/5.15/${qt_version}/single/qt-everywhere-opensource-src-${qt_version}.tar.xz}"
qt_sha256_default="efa99827027782974356aceff8a52bd3d2a8a93a54dd0db4cca41b5e35f1041c"
qt_sha256="${QT_SHA256:-${qt_sha256_default}}"

# 下载和构建缓存默认保存在仓库内，失败后可重试。
cache_dir="${QT_CACHE_DIR:-${script_dir}/build_cache}"
install_dir="${QT_INSTALL_DIR:-${repo_dir}/3rdparty/qt5}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

usage() {
    echo "Usage: $0 [--jobs N] [--cache DIR] [--install DIR]"
    echo "Environment overrides: JOBS, QT_URL, QT_SHA256, QT_CACHE_DIR, QT_SRC_DIR, QT_BUILD_DIR, QT_ARCHIVE, QT_INSTALL_DIR"
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

cache_dir="$(realpath -m -- "$cache_dir")"
install_dir="$(realpath -m -- "$install_dir")"
src_dir="$(realpath -m -- "$src_dir")"
build_dir="$(realpath -m -- "$build_dir")"
archive="$(realpath -m -- "$archive")"
if [[ "$src_dir" == "$build_dir" ]]; then
    echo "QT_SRC_DIR and QT_BUILD_DIR must be different directories" >&2
    exit 2
fi
if [[ -x "${install_dir}/bin/qmake" &&
      "$("${install_dir}/bin/qmake" -query QT_VERSION)" == "$qt_version" &&
      -f "${install_dir}/lib/cmake/Qt5Widgets/Qt5WidgetsConfig.cmake" &&
      -f "${install_dir}/lib/cmake/Qt5Concurrent/Qt5ConcurrentConfig.cmake" &&
      -f "${install_dir}/plugins/platforms/libqxcb.so" &&
      -x "${install_dir}/bin/lrelease" ]]; then
    echo "Qt ${qt_version} already installed at ${install_dir}; nothing to do."
    exit 0
fi
for command in tar sha256sum make awk perl python3 gcc g++ realpath pkg-config gperf; do
    command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done
if command -v curl >/dev/null; then
    downloader=(curl -fL --connect-timeout 30 --retry 3 --retry-delay 2 -C - -o)
elif command -v wget >/dev/null; then
    downloader=(wget --timeout=60 --tries=3 -c -O)
else
    echo "Missing downloader: install curl or wget" >&2
    exit 1
fi


trap 'echo "Qt build failed at line ${LINENO}: ${BASH_COMMAND}. Cache retained at ${cache_dir}" >&2' ERR

for package in xcb xcb-render xcb-renderutil xcb-shape xcb-randr xcb-xfixes \
    xcb-sync xcb-shm xcb-icccm xcb-keysyms xcb-image xcb-xkb xcb-xinerama xkbcommon-x11 \
    fontconfig freetype2; do
    if ! pkg-config --exists "$package"; then
        echo "Missing Qt/XCB development dependency: ${package} (see UI/README.md)" >&2
        exit 1
    fi
done

mkdir -p "${cache_dir}"
mkdir -p "${cache_dir}/tmp" "$(dirname -- "$archive")"
export TMPDIR="${cache_dir}/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
if [[ ! -f "${archive}" ]]; then
    echo "Downloading Qt ${qt_version} to ${cache_dir}..."
    downloaded=0
    for attempt in 1 2 3; do
        if "${downloader[@]}" "${archive}.part" "${qt_url}"; then
            downloaded=1
            break
        fi
        echo "Download attempt ${attempt} failed; retaining partial download for retry." >&2
    done
    [[ "$downloaded" == 1 ]] || { echo "Qt download failed" >&2; exit 1; }
    mv -- "${archive}.part" "$archive"
fi
actual_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${qt_sha256}" ]]; then
    echo "Qt archive SHA256 mismatch: expected ${qt_sha256}, got ${actual_sha256}" >&2
    exit 1
fi

# 规范解压目录，避免 configure 路径错误。
if [[ ! -f "${src_dir}/configure" ]]; then
    echo "Extracting Qt source..."
    mkdir -p "${src_dir}"
    tar -xJf "${archive}" -C "${src_dir}" --strip-components=1
fi

mkdir -p "${build_dir}"
pushd "${build_dir}" >/dev/null
configuration="${src_dir}|${install_dir}"
if [[ ! -f Makefile || ! -f .shape-match-qt-config ||
      "$(cat .shape-match-qt-config)" != "$configuration" ]]; then
    "${src_dir}/configure" \
        -prefix "${install_dir}" \
        -opensource -confirm-license \
        -release -no-opengl -xcb -recheck-all -no-feature-qdoc \
        -nomake examples -nomake tests \
        -skip qt3d -skip qtactiveqt -skip qtandroidextras -skip qtcharts \
        -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative \
        -skip qtgamepad -skip qtlocation -skip qtlottie -skip qtmultimedia \
        -skip qtnetworkauth -skip qtpurchasing -skip qtquick3d -skip qtremoteobjects \
        -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus \
        -skip qtserialport -skip qtspeech -skip qttranslations -skip qtwebengine \
        -skip qtvirtualkeyboard -skip qtwebchannel -skip qtwebglplugin \
        -skip qtwebsockets -skip qtwebview -skip qtwinextras -skip qtx11extras
    printf '%s\n' "$configuration" > .shape-match-qt-config
fi
make -j"${jobs}"
make install
popd >/dev/null

echo "Qt ${qt_version} installed at ${install_dir}."
