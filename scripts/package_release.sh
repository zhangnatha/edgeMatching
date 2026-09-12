#!/usr/bin/env bash
set -Eeuo pipefail

# 解析路径、构建发布版本并收集运行时依赖。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${script_dir}/.." && pwd -P)"
build_dir="${repo_dir}/build-release"
output_dir="${repo_dir}/dist/edgeMatching-linux"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

usage() {
    cat <<'EOF'
Usage: package_release.sh [--build-dir DIR] [--output DIR] [--jobs N] [--no-qt]

Builds a relocatable Linux release directory and a ZIP archive.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; build_dir="$2"; shift 2 ;;
        --output) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; output_dir="$2"; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; jobs="$2"; shift 2 ;;
        --no-qt) no_qt=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
no_qt="${no_qt:-0}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
command -v cmake >/dev/null || { echo "Missing required command: cmake" >&2; exit 1; }
# 自动在常见环境路径（如 anaconda、miniconda）中查找 patchelf
if ! command -v patchelf >/dev/null 2>&1; then
    for candidate in "$HOME/anaconda3/bin" "$HOME/miniconda3/bin" /opt/conda/bin; do
        if [[ -x "$candidate/patchelf" ]]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi
command -v patchelf >/dev/null 2>&1 || {
    echo "Missing required command: patchelf" >&2
    echo "Please install patchelf (e.g. 'sudo apt-get install patchelf' or 'conda install patchelf') or add it to PATH." >&2
    exit 1
}

# 防止误删源码、构建目录或根目录。
output_dir="$(realpath -m -- "$output_dir")"
case "$output_dir" in
    /|"$repo_dir"|"$build_dir"|"$build_dir"/*)
        echo "Refusing unsafe output directory: $output_dir" >&2; exit 2 ;;
    "$repo_dir"/*)
        case "$output_dir" in
            "$repo_dir/dist"|"$repo_dir/dist"/*) ;;
            *) echo "Refusing unsafe output directory: $output_dir" >&2; exit 2 ;;
        esac ;;
esac

qt_enabled=OFF
if [[ "$no_qt" == 0 && -f "$repo_dir/3rdparty/qt5/lib/cmake/Qt5/Qt5Config.cmake" ]]; then
    qt_enabled=ON
fi
echo "Configuring release build (Qt client: $qt_enabled)..."
cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_QT_CLIENT="$qt_enabled"
echo "Building release binaries..."
cmake --build "$build_dir" --parallel "$jobs"

stage="$(mktemp -d "${TMPDIR:-/tmp}/edgeMatching-release.XXXXXX")"
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT
cmake --install "$build_dir" --prefix "$stage"
mkdir -p "$stage/lib"

copy_matches() {
    local source_dir="$1" pattern="$2"
    [[ -d "$source_dir" ]] || return 0
    find "$source_dir" -maxdepth 1 \( -type f -o -type l \) -name "$pattern" \
        -exec cp -a -- {} "$stage/lib/" \;
}

# 安装步骤已经复制核心库；补充仓库内 OpenCV 和 Qt 的动态库。
for opencv_module in core imgproc imgcodecs highgui calib3d; do
    copy_matches "$repo_dir/3rdparty/opencv/lib" "libopencv_${opencv_module}.so*"
done
if [[ "$qt_enabled" == ON ]]; then
    copy_matches "$repo_dir/3rdparty/qt5/lib" 'libQt5*.so*'
    mkdir -p "$stage/plugins"
    for plugin_dir in platforms imageformats platformthemes platforminputcontexts xcbglintegrations iconengines; do
        if [[ -d "$repo_dir/3rdparty/qt5/plugins/$plugin_dir" ]]; then
            cp -a "$repo_dir/3rdparty/qt5/plugins/$plugin_dir" "$stage/plugins/"
        fi
    done
    cat <<'EOF' > "$stage/bin/qt.conf"
[Paths]
Prefix = ..
Plugins = plugins
Libraries = lib
EOF
fi

# 使用相对 RPATH，使发布目录不依赖源码或构建目录。
while IFS= read -r -d '' binary; do
    patchelf --set-rpath '$ORIGIN/../lib' "$binary"
done < <(find "$stage/bin" -maxdepth 1 -type f -perm -u+x -print0)
while IFS= read -r -d '' library; do
    patchelf --set-rpath '$ORIGIN' "$library"
done < <(find "$stage/lib" -maxdepth 1 -type f -name '*.so*' -print0)
if [[ -d "$stage/plugins" ]]; then
    while IFS= read -r -d '' plugin; do
        patchelf --set-rpath '$ORIGIN/../../lib' "$plugin"
    done < <(find "$stage/plugins" -type f -name '*.so*' -print0)
fi

mkdir -p "$stage/docs"
cp -a "$repo_dir/README.md" "$repo_dir/LICENSE" "$stage/"
cp -a "$repo_dir/docs/template_matching_algorithm.md" "$stage/docs/"
printf 'edgeMatching release package\nBuilt from: %s\nQt client: %s\n' "$repo_dir" "$qt_enabled" \
    > "$stage/RELEASE.txt"

mkdir -p "$(dirname -- "$output_dir")"
rm -rf -- "$output_dir"
mv -- "$stage" "$output_dir"
trap - EXIT
rm -f -- "${output_dir}.zip"
(cd "$(dirname -- "$output_dir")" && zip -qr "$(basename -- "$output_dir").zip" "$(basename -- "$output_dir")")
echo "Release package created at $output_dir"
echo "ZIP archive created at ${output_dir}.zip"
