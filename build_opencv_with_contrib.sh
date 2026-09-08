#!/usr/bin/env bash

set -Eeuo pipefail

# 可配置变量
readonly OPENCV_VERSION=4.7.0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly INSTALL_DIR="${SCRIPT_DIR}/3rdparty/opencv"

# 下载、解压和编译都在独立临时目录中进行，避免污染项目目录。
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/edgeMatching-opencv.XXXXXX")"
readonly WORK_DIR
readonly OPENCV_DIR="${WORK_DIR}/opencv-${OPENCV_VERSION}"
readonly CONTRIB_DIR="${WORK_DIR}/opencv_contrib-${OPENCV_VERSION}"
readonly BUILD_DIR="${WORK_DIR}/build"

cleanup() {
    local exit_code=$?
    trap - EXIT
    if [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR}" ]]; then
        echo "🧹 清理临时目录 ${WORK_DIR}"
        rm -rf -- "${WORK_DIR}"
    fi
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# 下载源码
echo "📦 下载 OpenCV ${OPENCV_VERSION} ..."
wget -O "${WORK_DIR}/opencv.zip" \
    "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
unzip -q "${WORK_DIR}/opencv.zip" -d "${WORK_DIR}"

echo "📦 下载 opencv_contrib ${OPENCV_VERSION} ..."
wget -O "${WORK_DIR}/opencv_contrib.zip" \
    "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip"
unzip -q "${WORK_DIR}/opencv_contrib.zip" -d "${WORK_DIR}"

# 准备构建目录
mkdir -p "${BUILD_DIR}"

echo "⚙️ 配置 CMake 编译选项..."
cmake -S "${OPENCV_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DOPENCV_EXTRA_MODULES_PATH="${CONTRIB_DIR}/modules" \
  -DWITH_OPENMP=ON \
  -DWITH_TBB=OFF \
  -DWITH_IPP=ON \
  -DWITH_EIGEN=ON \
  -DWITH_QT=OFF \
  -DWITH_GTK=ON \
  -DWITH_V4L=ON \
  -DWITH_OPENCL=OFF \
  -DWITH_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_world=OFF

echo "🔨 编译 OpenCV + contrib..."
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo "📥 安装到 ${INSTALL_DIR}"
cmake --install "${BUILD_DIR}"

echo "✅ OpenCV ${OPENCV_VERSION} + opencv_contrib 编译安装完成！"
