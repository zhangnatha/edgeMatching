#!/bin/bash

set -e

# 可配置变量
OPENCV_VERSION=4.7.0
INSTALL_DIR=$(pwd)/3rdparty/opencv
OPENCV_DIR=opencv-${OPENCV_VERSION}
CONTRIB_DIR=opencv_contrib-${OPENCV_VERSION}
BUILD_DIR=opencv_build

# 下载源码
if [ ! -d "${OPENCV_DIR}" ]; then
    echo "📦 下载 OpenCV ${OPENCV_VERSION} ..."
    wget -O ${OPENCV_DIR}.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
    unzip -q ${OPENCV_DIR}.zip
fi

if [ ! -d "${CONTRIB_DIR}" ]; then
    echo "📦 下载 opencv_contrib ${OPENCV_VERSION} ..."
    wget -O ${CONTRIB_DIR}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
    unzip -q ${CONTRIB_DIR}.zip
fi

# 准备构建目录
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "⚙️ 配置 CMake 编译选项..."
cmake ../${OPENCV_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DOPENCV_EXTRA_MODULES_PATH=../${CONTRIB_DIR}/modules \
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
make -j$(nproc)

echo "📥 安装到 ${INSTALL_DIR}"
make install

cd .. && rm -rf ${CONTRIB_DIR}.zip ${OPENCV_DIR}.zip
echo "✅ OpenCV ${OPENCV_VERSION} + opencv_contrib 编译安装完成！"

