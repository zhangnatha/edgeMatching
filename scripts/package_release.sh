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
for tool in cmake zip ldd realpath; do
    command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done
# 优先使用仓库内的新版本，再检测系统和常见 Conda 环境。
if [[ -x "$repo_dir/3rdparty/tools/bin/patchelf" ]]; then
    export PATH="$repo_dir/3rdparty/tools/bin:$PATH"
elif ! command -v patchelf >/dev/null 2>&1; then
    for candidate in "$HOME/anaconda3/bin" "$HOME/miniconda3/bin" /opt/conda/bin; do
        if [[ -x "$candidate/patchelf" ]]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi
command -v patchelf >/dev/null 2>&1 || {
    echo "Missing required command: patchelf" >&2
    echo "Run bash scripts/build_patchelf.sh or install patchelf >= 0.14." >&2
    exit 1
}
patchelf_version="$(patchelf --version | awk '{print $2}')"
if [[ "$(printf '%s\n' 0.14 "$patchelf_version" | sort -V | head -n 1)" != 0.14 ]]; then
    echo "patchelf ${patchelf_version} is too old; run bash scripts/build_patchelf.sh (requires >= 0.14)." >&2
    exit 1
fi

# 防止误删源码、构建目录或根目录。
build_dir="$(realpath -m -- "$build_dir")"
output_dir="$(realpath -m -- "$output_dir")"
case "$build_dir" in
    "$output_dir"/*) echo "Refusing output directory containing the build directory: $output_dir" >&2; exit 2 ;;
esac
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
qt_args=()
if [[ "$no_qt" == 0 ]]; then
    [[ -f "$repo_dir/3rdparty/qt5/lib/cmake/Qt5/Qt5Config.cmake" ]] || {
        echo "Local Qt5 not found. Run UI/build_qt5.sh or use --no-qt." >&2; exit 1;
    }
    qt_enabled=ON
    qt_args=("-DQt5_DIR=$repo_dir/3rdparty/qt5/lib/cmake/Qt5" "-DCMAKE_PREFIX_PATH=$repo_dir/3rdparty/qt5")
fi
echo "Configuring release build (Qt client: $qt_enabled)..."
cmake -S "$repo_dir" -B "$build_dir" \
    -U 'Qt5*' \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_QT_CLIENT="$qt_enabled" "${qt_args[@]}"
echo "Building release binaries..."
cmake --build "$build_dir" --parallel "$jobs"

stage="$(mktemp -d "$repo_dir/edgeMatching-release.XXXXXX")"
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

# 收集可执行文件、核心库及 Qt 插件的完整依赖闭包。
# glibc 和动态加载器由目标系统提供，其余依赖随发布包携带。
dependency_library_path="$stage/lib:$repo_dir/3rdparty/opencv/lib:$repo_dir/3rdparty/qt5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
changed=1
while [[ "$changed" == 1 ]]; do
    changed=0
    while IFS= read -r -d '' binary; do
        dependencies="$(LD_LIBRARY_PATH="$dependency_library_path" ldd "$binary")"
        if [[ "$dependencies" == *'not found'* ]]; then
            echo "Unresolved dependencies for $binary:" >&2
            echo "$dependencies" >&2
            exit 1
        fi
        while IFS= read -r dependency; do
            name="$(basename -- "$dependency")"
            case "$name" in
                libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libanl.so.*|libnss_*.so.*|ld-linux*.so.*) continue ;;
            esac
            if [[ ! -e "$stage/lib/$name" ]]; then
                cp -L -- "$dependency" "$stage/lib/$name"
                changed=1
            fi
        done < <(printf '%s\n' "$dependencies" | sed -n 's/^[[:space:]]*[^[:space:]]* => \(.*\) (0x[[:xdigit:]]*).*$/\1/p')
    done < <(find "$stage/bin" "$stage/lib" "$stage/plugins" -type f \( -name '*.so*' -o -perm -u+x \) -print0 2>/dev/null)
done

# 使用相对 RPATH，使发布目录不依赖源码或构建目录。
while IFS= read -r -d '' binary; do
    patchelf --set-rpath '$ORIGIN/../lib' "$binary"
done < <(find "$stage/bin" -maxdepth 1 -type f -perm -u+x -print0)
while IFS= read -r -d '' library; do
    patchelf --set-rpath '$ORIGIN' "$library" || {
        echo "Failed to patch library: $library" >&2; exit 1;
    }
done < <(find "$stage/lib" -maxdepth 1 -type f -name '*.so*' -print0)
if [[ -d "$stage/plugins" ]]; then
    while IFS= read -r -d '' plugin; do
        patchelf --set-rpath '$ORIGIN/../../lib' "$plugin"
    done < <(find "$stage/plugins" -type f -name '*.so*' -print0)
fi

mkdir -p "$stage/docs"
cp -a "$repo_dir/README.md" "$repo_dir/LICENSE" "$stage/"
cp -a "$repo_dir/docs/template_matching_algorithm.md" "$stage/docs/"
if [[ -d "$repo_dir/assert/.md" ]]; then
    mkdir -p "$stage/assert/.md"
    cp -a "$repo_dir/assert/.md/." "$stage/assert/.md/"
fi
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
