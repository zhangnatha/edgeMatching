# Linux 构建、验证与打包 SOP

适用于 Ubuntu 18.04 及更高版本。默认从源码将 OpenCV 4.7.0（含 contrib）和 Qt 5.15.16 安装到仓库的 `3rdparty`，无需安装系统 Qt。以下命令均在仓库根目录执行。

## 1. 获取源码与准备环境

```bash
git clone https://github.com/zhangnatha/edgeMatching.git
cd edgeMatching
sudo apt update
sudo apt install build-essential pkg-config git wget curl unzip zip xz-utils \
  perl python3 gperf bison flex libgtk-3-dev libeigen3-dev \
  libfontconfig1-dev libfreetype6-dev libdbus-1-dev libglib2.0-dev \
  libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev libxrender-dev \
  libxcb1-dev libxcb-render0-dev libxcb-render-util0-dev libxcb-shape0-dev \
  libxcb-randr0-dev libxcb-xfixes0-dev libxcb-sync-dev libxcb-shm0-dev \
  libxcb-icccm4-dev libxcb-keysyms1-dev libxcb-image0-dev libxcb-xkb-dev \
  libxcb-xinerama0-dev libxcb-util-dev libxkbcommon-dev libxkbcommon-x11-dev
cmake --version
python3 --version
```

需要 CMake 3.16+ 和 Python 3.6+。Ubuntu 18.04 默认 CMake 3.10 不满足要求，应先安装新版 CMake；较新 Ubuntu 可使用 `sudo apt install cmake`，并检查实际版本。编译器需要支持 C++11；OpenMP 为可选依赖。

为让发布包支持 Ubuntu 18.04 及更新版本，应在 Ubuntu 18.04 或具有相同最低 glibc 基线的环境中构建。较新系统编译出的二进制不能据此保证可在旧系统运行。

## 2. 编译 OpenCV 与本地 Qt

```bash
bash build_opencv_with_contrib.sh
bash UI/build_qt5.sh --jobs 8
```

OpenCV 下载、解压、编译及临时文件位于仓库根目录 `edgeMatching-opencv.XXXXXX`，不使用 `/tmp`；成功后清理工作目录，失败后保留。安装结果位于 `3rdparty/opencv`。

Qt 下载及构建缓存位于 `UI/build_cache`，安装结果位于 `3rdparty/qt5`。脚本校验 Qt 源码 SHA256，保留缓存供重试，构建 Widgets、Concurrent、XCB 插件及 `lrelease`。Qt 默认关闭 OpenGL，跳过 WebEngine 等模块。

可用 `--cache DIR`、`--install DIR` 指定 Qt 路径；下载镜像可通过 `QT_URL` 设置。更多变量见 [Qt 客户端说明](../UI/README.md)。并发任务数应按内存调整；OpenCV 脚本使用 `nproc`。

## 3. 编译项目与验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -U 'Qt5*' \
  -DOpenCV_DIR="$PWD/3rdparty/opencv/lib/cmake/opencv4" \
  -DQt5_DIR="$PWD/3rdparty/qt5/lib/cmake/Qt5" \
  -DBUILD_QT_CLIENT=ON
cmake --build build --parallel 8
(cd build && ctest --output-on-failure)
```

生成 `build/train`、`build/inference` 和 `build/UI/shape_match_qt`。CTest 的 `assert_matrix` 验证 README 中 11 次训练和 26 次推理的数量与模板 ID 分布；`qt_client_startup` 验证 Qt 启动及语言切换。

单独运行 CLI 样例矩阵：

```bash
python3 scripts/verify_examples.py --build-dir build
```

切换 Qt 安装时使用新构建目录，或像上面一样清除 `Qt5*` 缓存并明确指定 `Qt5_DIR`。仅编译 CLI 可使用：

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DBUILD_QT_CLIENT=OFF \
  -DOpenCV_DIR="$PWD/3rdparty/opencv/lib/cmake/opencv4"
cmake --build build-core --parallel 8
(cd build-core && ctest --output-on-failure)
```

## 4. 运行样例与客户端

```bash
./build/train assert/m8.bmp --id 8 --output build/model_8.json
./build/inference assert/src8.bmp build/model_8.json \
  --min-score 0.95 --min-visible-ratio 0.5 --subpixel \
  --output build/src8.result.png
./build/UI/shape_match_qt
```

`src8` 预期匹配 7 个实例，结果图和同名 JSON 写入 `build`。GUI 需要可用的桌面显示环境；无显示环境仅检查启动可执行：

```bash
QT_QPA_PLATFORM=offscreen ./build/UI/shape_match_qt --smoke-test
```

需要观察匹配中间过程时，另建目录开启可视化：

```bash
cmake -S . -B build-debug -DSHAPE_MATCH_VISUALIZE_COARSE=ON \
  -DSHAPE_MATCH_VISUALIZE_FINE=ON
cmake --build build-debug --parallel 8
```

AVX2 通过运行时 CPU 检测启用，无需添加全局 `-mavx2`；CLI 的 `--simd` 在不支持 AVX2 时回退标量路径。

## 5. 安装与生成发布包

安装项目自身的目标：

```bash
cmake --install build --prefix build/publish
```

上述安装不等于完整运行依赖打包。需要可复制的发布目录和 ZIP 时执行：

```bash
bash scripts/build_patchelf.sh
bash scripts/package_release.sh --jobs 8 --output dist/edgeMatching-linux
```

打包要求 CMake 3.16+、`zip` 和 patchelf 0.14+。`build_patchelf.sh` 校验并编译 patchelf 0.14.3，安装到 `3rdparty/tools`，无需 root；发布脚本优先使用本地版本。Ubuntu 18.04 默认 patchelf 0.9 不满足要求。

发布脚本默认使用 `3rdparty/qt5`，收集项目、OpenCV、Qt 插件以及非 glibc 动态依赖，并设置相对 RPATH。默认构建目录为 `build-release`，暂存目录位于仓库根目录。可用 `--build-dir DIR`、`--output DIR`、`--jobs N` 修改路径和并发数。仅发布 CLI：

```bash
bash scripts/package_release.sh --jobs 8 --no-qt \
  --output dist/edgeMatching-linux-cli
```

输出 `dist/edgeMatching-linux` 目录和 `dist/edgeMatching-linux.zip`。将整个目录复制或解压到目标机器，从 `bin` 启动：

```bash
./dist/edgeMatching-linux/bin/shape_match_qt --smoke-test
./dist/edgeMatching-linux/bin/shape_match_qt
```

这是目录式便携安装，不生成 deb/rpm；目标机器仍需要匹配的 glibc 和桌面显示环境。不要只复制可执行文件。发布包不包含 `assert/*.bmp`，样例验收需另从源码准备输入图片。

## 6. 验收范围

已在 Ubuntu 18.04 完成源码编译、样例矩阵与 GUI 启动验证；发布包也已在 Ubuntu 22.04/24.04 隔离用户空间中验证。隔离用户空间共用宿主内核，不等同于各版本完整桌面实机测试。

持续集成配置见 [build.yml](../.github/workflows/build.yml)，包含 Ubuntu 22.04/24.04 的依赖构建、样例、Qt 启动及打包验证任务。发布前应在目标架构和最低支持系统上验收。
