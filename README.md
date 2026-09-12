# edgeMatching

[![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04%2B-E95420)](https://ubuntu.com/)[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6)](https://www.microsoft.com/windows)[![C%2B%2B](https://img.shields.io/badge/C%2B%2B-11-blue)](https://isocpp.org/)[![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.10-064F8C)](https://cmake.org/)[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-green)](https://opencv.org/)[![Qt](https://img.shields.io/badge/Qt-5.x-41CD52)](https://www.qt.io/)[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

基于梯度方向余弦相似度的工业边缘形状模板匹配。项目包含 C++ 核心库、
`train`/`inference` 命令行程序，以及支持中英文的 Qt5 客户端。

## 功能

- 多层金字塔、旋转、尺度和多模板搜索。
- Canny 像素级、Canny + 抛物线亚像素、Devernay 亚像素三种训练边缘后端。
- 亚像素位姿精修、部分可见目标、全局/局部极性忽略、轮廓支撑带 NMS。
- 可选 AVX2 SIMD；运行时检测 CPU，不支持时自动使用可移植标量实现。
- Qt 客户端提供训练/推理参数页、金字塔特征预览、结果图及中英文切换。

核心算法说明见 [docs/template_matching_algorithm.md](docs/template_matching_algorithm.md)，
Qt 专项说明见 [UI/README.md](UI/README.md)。

## 环境与依赖

支持平台与必需依赖：

- Ubuntu 18.04 及更高版本：支持 C++11 的 GCC/Clang、CMake 3.10 或更高版本。
- Windows 10/11：Visual Studio 2019/2022（MSVC）或 MinGW-w64、CMake 3.10 或更高版本。
- OpenCV 4.7.0，使用 `core`、`imgproc`、`highgui`、`imgcodecs`、`calib3d`。
  将预编译包放在 `3rdparty/opencv`，或通过 `-DOpenCV_DIR=...` 指向 OpenCV 的
  `OpenCVConfig.cmake` 所在目录。Ubuntu 可运行 `./build_opencv_with_contrib.sh`；
  Windows 请使用 OpenCV 官方 Windows 包或从源码用 CMake 构建。

可选：

- Qt5 Widgets 和 Concurrent。顶层 CMake 默认构建 Qt 客户端，并优先使用
  `3rdparty/qt5`；Windows 可用 Qt Online Installer 安装后通过 `-DCMAKE_PREFIX_PATH=...`
  指定。没有 Qt 时使用 `-DBUILD_QT_CLIENT=OFF`。
- AVX2 仅在 x86 且 CPU 支持时启用；不应为 ARM 或不支持 AVX2 的平台添加
  `-mavx2` 全局编译选项。

如果尚未安装 Qt5，可执行：

```bash
UI/build_qt5.sh --jobs 8
```

脚本下载并校验 Qt 5.15.16，将最小模块安装到 `3rdparty/qt5`；也可用
`--cache`、`--install` 或对应环境变量覆盖路径。构建 Qt 所需的系统开发包
（X11/XCB、fontconfig、freetype、ICU 等）需预先由系统包管理器安装。

Ubuntu 18.04 的源码构建依赖可一次安装：

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev libomp-dev \
  pkg-config patchelf zip
```

若使用仓库内 Qt5，仍需安装 Qt 构建所需的 X11/XCB、fontconfig、freetype 和 ICU
开发包；`UI/build_qt5.sh` 会将 Qt 安装到 `3rdparty/qt5`。为保证发布包能在 Ubuntu
18.04 及更高版本运行，Linux 发布包应在 Ubuntu 18.04 或兼容的最低 glibc 环境中
构建，再复制到更新版本系统。

Windows 源码构建需要 Git、CMake、Visual Studio 2019/2022（含 C++ 桌面组件）或
MinGW-w64，以及 OpenCV 4.7.0 的 Windows 开发包。Qt 客户端还需要 Qt5 Widgets/
Concurrent 和同版本的 `windeployqt.exe`；将 OpenCV `bin` 目录加入 `PATH`，或由
Windows 发布脚本复制 DLL 到发布目录。

## 编译、验证和安装

从仓库根目录执行，一次配置同时生成核心库、CLI 和 Qt 客户端：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

生成文件：

```text
build/train
build/inference
build/UI/shape_match_qt       # BUILD_QT_CLIENT=ON 且找到 Qt5 时
```

无 Qt 环境的核心构建：

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DBUILD_QT_CLIENT=OFF
cmake --build build-core --parallel
```

安装到 CMake 构建目录下的 `publish`：

```bash
cmake --install build --prefix build/publish
```

Windows PowerShell（Visual Studio 生成器）使用同一套 CMake 工程：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR="$PWD\3rdparty\opencv\build"
cmake --build build --config Release
cmake --install build --config Release --prefix build\publish
```

如果使用 MinGW，将生成器替换为 `-G "MinGW Makefiles"`，并使用
`cmake --build build --config Release --parallel`。运行时将
`3rdparty\opencv\bin` 中的 DLL 放在可执行文件同目录或加入 `PATH`；Qt 客户端
还需运行 `windeployqt build\UI\Release\shape_match_qt.exe`。

Linux/Windows 的无源码发布包由以下脚本生成，包内包含可执行文件、核心库、依赖
动态库和 Qt 插件（目标平台可用时）：

```bash
./scripts/package_release.sh --output dist/edgeMatching-linux
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1 -OutputDir .\dist\edgeMatching-windows
```

脚本会在失败时退出，不删除源码或已有构建目录；详细参数见脚本的 `--help` 或
`Get-Help` 输出。

Linux 打包依赖 `cmake`、`patchelf` 和 `zip` 工具。针对不同运行环境的依赖准备方式如下：

- **安装了 Conda（Anaconda / Miniconda）**：
  - 若 Conda 环境中已带有 `patchelf`，`package_release.sh` 脚本已内置对 `~/anaconda3/bin`、`~/miniconda3/bin` 等常见路径的自动探测，可直接执行脚本；也可在终端先执行 `conda activate`。
  - 若 Conda 环境中缺少该工具，无需 root / `sudo` 权限即可直接在当前环境中安装：
    ```bash
    conda install -c conda-forge patchelf zip
    ```
- **未安装 Conda（纯系统环境）**：
  - **有 sudo 权限**：直接通过系统包管理器安装：
    ```bash
    sudo apt install -y patchelf zip
    ```
  - **无 sudo 权限（普通用户）**：可直接下载 GitHub 发布的独立静态二进制（单文件免编译安装）并加入 PATH：
    ```bash
    mkdir -p ~/.local/bin
    curl -sSL https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz | tar -xz -C ~/.local/bin patchelf
    export PATH="$HOME/.local/bin:$PATH"
    ```

匹配过程可视化是独立的 CMake 选项，默认关闭：

```bash
cmake -S . -B build-debug -DSHAPE_MATCH_VISUALIZE_COARSE=ON \
  -DSHAPE_MATCH_VISUALIZE_FINE=ON
cmake --build build-debug --parallel
```

## CLI：训练模板

```text
build/train [template_image] [--id N]
            [--edge-method pixel|current|devernay]
            [--output FILE] [--pyramid-output FILE]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `template_image` | `../assert/m1.png` | 8 位灰度或彩色模板图；彩色图在核心内部转灰度。 |
| `--id N` | `1` | 正整数模板 ID。 |
| `--edge-method` | `current` | `pixel`：Canny 像素级；`current`：Canny + 抛物线亚像素；`devernay`：Devernay 亚像素。 |
| `--output FILE` | 可执行文件同级的 `<模板文件名>.json` | JSON 输出路径；扩展名由调用者指定。 |
| `--pyramid-output FILE` | 不保存 | 保存各层灰度图和 canonical 特征叠加预览。 |

CLI 当前未暴露的训练默认值为：金字塔层数 `-1`（自动）、角度 `-180..180°`、
步长 `1°`、Otsu 关闭、最小/最大对比度 `25/100`。Qt 训练页可以直接设置这些值。
模型只保存 canonical 特征，加载时按推理角度生成缓存。

示例：

```bash
build/train assert/m9_1.bmp --id 10 \
  --edge-method devernay \
  --output build/m9_1.json \
  --pyramid-output build/m9_1.pyramid.png
```

## CLI：推理

```text
build/inference [search_image] [model1.json model2.json ...]
                [--min-score N] [--max-overlap N]
                [--angle-start DEG] [--angle-end DEG]
                [--scale-min N] [--scale-max N] [--scale-step N]
                [--min-visible-ratio N] [--min-contrast N]
                [--metric use-polarity|ignore-global-polarity|ignore-local-polarity]
                [--subpixel] [--simd] [--output FILE]
```

| 参数 | 默认值 | 合法范围/说明 |
| --- | --- | --- |
| `search_image` | `../assert/src.bmp` | 待测图像，CLI 以 `IMREAD_GRAYSCALE` 读取。 |
| `model1...` | `./model.json` | 一个或多个 JSON/BIN 模型；多模型 ID 重复或非正时仅在内存中确定性重编号，不修改源文件。 |
| `--min-score` | `0.7` | `[0,1]`，最终最低匹配得分。 |
| `--max-overlap` | `0.5` | `[0,1]`，重叠抑制阈值。 |
| `--angle-start/end` | `-180/180` | 整数角度，`-180 ≤ start ≤ end ≤ 180`；终止角是绝对角度，不是跨度。 |
| `--scale-min/max/step` | `1/1/1` | `0 < min ≤ max`，步长为正；三个值为 1 时为单尺度。 |
| `--min-visible-ratio` | `1.0` | `[0,1]`；边界截断目标可降低，例如 `0.5`。 |
| `--min-contrast` | `0` | `[0,361]` 的整数；过滤待测图弱梯度。 |
| `--metric` | `use-polarity` | 有符号极性、忽略全局极性或逐点忽略局部极性。 |
| `--subpixel` | 关闭 | NMS 后精修 `x/y/angle/scale`；不开启时仍可使用亚像素模板特征。 |
| `--simd` | 关闭 | 请求 AVX2；CPU 不支持时自动回退标量路径。 |
| `--output` | `result.png` | 保存带核心绘制结果的图像；同时在相同目录写入同名 `.json`。 |

CLI 的固定搜索默认值为：最大结果数 `200`、搜索层数 `-1`、贪婪度 `0.9`、
按 Y 排序、全图 ROI。示例：

```bash
build/inference assert/src8.bmp build/model_8.json \
  --min-score 0.95 --min-visible-ratio 0.5 \
  --subpixel --output build/src8.result.png
```

输出行格式：

```text
[0] template_id=8 x=... y=... angle=... score=... scale=... visible=... matched=...
```

`visible` 是几何可见特征比例，`matched` 是可见点中通过对比度和梯度检查的比例。

## Qt5 客户端

```bash
./build/UI/shape_match_qt
```

客户端调用与 CLI 相同的 `CreateTemplate`、`SearchTemplate` 和绘制接口，不另有一套
匹配算法。文件选择默认打开 `assert`（模板/输入图像）或可执行文件目录（模型）；
模型、金字塔预览、结果 PNG 和同名结果 JSON 默认写入 `shape_match_qt` 同级目录。
结果 PNG 是核心库绘制后的文件，客户端窗口显示则使用独立的矢量叠加层。

训练页和推理页参数对应关系如下：

| Qt 参数 | 核心/CLI 对应 | 默认值 |
| --- | --- | --- |
| 模板图像、Template ID | `template_image`、`--id` | `../assert/m1.png`、`1` |
| 金字塔层数、角度起止/步长 | `createTemplate` 的 `num_levels`、`angle_start/end/step` | `-1`、`-180/180/1` |
| Otsu、最小/最大对比度 | `create_otsu`、`min/max_contrast` | 关、`25/100` |
| 边缘特征算法 | `--edge-method` / `EdgeMethod` | Canny + 抛物线亚像素 |
| 输入图像、模型（可多选） | `search_image`、`model1...` | 无；需选择 |
| ROI x/y/w/h | `SearchTemplate` ROI 重载 | `0/0/0/0`（全图） |
| 搜索角度起止、最小得分、最大重叠 | `--angle-start/end`、`--min-score`、`--max-overlap` | `-180/180`、`0.7`、`0.5` |
| 最大匹配数、搜索层数、贪婪度、按 Y 排序 | 核心 API 参数（CLI 固定为 `200/-1/0.9/true`） | `200/-1/0.9/开` |
| 尺度最小/最大/步长、最小可见比例 | `--scale-min/max/step`、`--min-visible-ratio` | `1/1/1`、`1` |
| 搜索最小对比度、匹配度量 | `--min-contrast`、`--metric` | `0`、`use-polarity` |
| 亚像素精修、AVX2 SIMD | `--subpixel`、`--simd` | 关、关 |

要让 Qt 与矩阵中的 CLI 结果对齐，训练时使用相同图像、ID、边缘算法和训练参数，
推理时使用相同模型、图像及 CLI 参数；Qt 的最大匹配数设为 `200`、搜索层数设为
`-1`、贪婪度设为 `0.9`、按 Y 排序开启、ROI 四项为 0。Qt 日志会输出实际参数和
等价 `inference` 命令，训练/推理纯耗时也分别显示。
CLI 接受的搜索角度为整数 `[-180,180]`；Qt 控件范围更宽时，跨端对比仍应使用该
公共范围（CLI 会拒绝范围外的角度）。

## assert 全量回归矩阵

该矩阵使用仓库 `assert/` 中的 11 个模板和 26 张待测图。下表命令直接调用
`build/train` 和 `build/inference`，不依赖额外测试二进制，也不调用 CTest；
因此可独立逐案例复现。表中的期望分布来自删除测试目录前的全量回归结果，
格式为 `template_id:数量`。

先编译核心程序，并创建独立输出目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
mkdir -p build/assert_matrix
```

### 训练命令

以下 11 条命令生成与矩阵相同的模型。每条命令同时保存 JSON 模型和各层金字塔预览图。

| 模板 ID | 模板图像 | 可复制训练命令 |
| ---: | --- | --- |
| 1 | `assert/m1.png` | `build/train assert/m1.png --id 1 --output build/assert_matrix/model_1.json --pyramid-output build/assert_matrix/pyramid_1.png` |
| 2 | `assert/m2.png` | `build/train assert/m2.png --id 2 --output build/assert_matrix/model_2.json --pyramid-output build/assert_matrix/pyramid_2.png` |
| 3 | `assert/m3.png` | `build/train assert/m3.png --id 3 --output build/assert_matrix/model_3.json --pyramid-output build/assert_matrix/pyramid_3.png` |
| 4 | `assert/m4.bmp` | `build/train assert/m4.bmp --id 4 --output build/assert_matrix/model_4.json --pyramid-output build/assert_matrix/pyramid_4.png` |
| 5 | `assert/m5.jpg` | `build/train assert/m5.jpg --id 5 --output build/assert_matrix/model_5.json --pyramid-output build/assert_matrix/pyramid_5.png` |
| 6 | `assert/m6.bmp` | `build/train assert/m6.bmp --id 6 --output build/assert_matrix/model_6.json --pyramid-output build/assert_matrix/pyramid_6.png` |
| 7 | `assert/m7.bmp` | `build/train assert/m7.bmp --id 7 --output build/assert_matrix/model_7.json --pyramid-output build/assert_matrix/pyramid_7.png` |
| 8 | `assert/m8.bmp` | `build/train assert/m8.bmp --id 8 --output build/assert_matrix/model_8.json --pyramid-output build/assert_matrix/pyramid_8.png` |
| 9 | `assert/m9.bmp` | `build/train assert/m9.bmp --id 9 --output build/assert_matrix/model_9.json --pyramid-output build/assert_matrix/pyramid_9.png` |
| 10 | `assert/m9_1.bmp` | `build/train assert/m9_1.bmp --id 10 --output build/assert_matrix/model_10.json --pyramid-output build/assert_matrix/pyramid_10.png` |
| 11 | `assert/m9_2.bmp` | `build/train assert/m9_2.bmp --id 11 --output build/assert_matrix/model_11.json --pyramid-output build/assert_matrix/pyramid_11.png` |

### 推理命令矩阵

每行推理命令都保存绘制后的 PNG，并由 CLI 在相同目录自动保存同名 JSON。执行前应先完成
上表 11 个训练命令。`src5=161` 是重复结构回归基线，不单独宣称 161 个均为业务真阳性。

| 案例 | 可复制推理命令 | 期望结果数 | 期望 `template_id` 分布 |
| --- | --- | ---: | --- |
| `src1_2_3` | `build/inference assert/src1_2_3.bmp build/assert_matrix/model_1.json build/assert_matrix/model_2.json build/assert_matrix/model_3.json --angle-start -5 --angle-end 5 --min-visible-ratio 0.5 --output build/assert_matrix/result_src1_2_3.png` | 33 | `1:12 2:9 3:12` |
| `src4` | `build/inference assert/src4.bmp build/assert_matrix/model_4.json --output build/assert_matrix/result_src4.png` | 3 | `4:3` |
| `src5` | `build/inference assert/src5.bmp build/assert_matrix/model_5.json --min-score 0.65 --output build/assert_matrix/result_src5.png` | 161 | `5:161` |
| `src6` | `build/inference assert/src6.jpg build/assert_matrix/model_6.json --output build/assert_matrix/result_src6.png` | 15 | `6:15` |
| `src7_1` | `build/inference assert/src7_1.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_1.png` | 1 | `7:1` |
| `src7_2` | `build/inference assert/src7_2.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_2.png` | 1 | `7:1` |
| `src7_3` | `build/inference assert/src7_3.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_3.png` | 1 | `7:1` |
| `src7_4` | `build/inference assert/src7_4.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_4.png` | 1 | `7:1` |
| `src7_5` | `build/inference assert/src7_5.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_5.png` | 1 | `7:1` |
| `src7_6` | `build/inference assert/src7_6.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_6.png` | 1 | `7:1` |
| `src7_7` | `build/inference assert/src7_7.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_7.png` | 1 | `7:1` |
| `src7_8` | `build/inference assert/src7_8.bmp build/assert_matrix/model_7.json --output build/assert_matrix/result_src7_8.png` | 1 | `7:1` |
| `src8` | `build/inference assert/src8.bmp build/assert_matrix/model_8.json --min-score 0.95 --min-visible-ratio 0.5 --output build/assert_matrix/result_src8.png` | 7 | `8:7` |
| `src9_1` | `build/inference assert/src9_1.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_1.png` | 4 | `9:2 10:2` |
| `src9_2` | `build/inference assert/src9_2.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_2.png` | 8 | `9:4 10:3 11:1` |
| `src9_3` | `build/inference assert/src9_3.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_3.png` | 5 | `10:4 11:1` |
| `src9_4` | `build/inference assert/src9_4.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.890 --output build/assert_matrix/result_src9_4.png` | 4 | `9:2 10:1 11:1` |
| `src9_5` | `build/inference assert/src9_5.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_5.png` | 3 | `9:3` |
| `src9_6` | `build/inference assert/src9_6.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.800 --output build/assert_matrix/result_src9_6.png` | 4 | `9:3 11:1` |
| `src9_7` | `build/inference assert/src9_7.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_7.png` | 5 | `9:2 10:2 11:1` |
| `src9_8` | `build/inference assert/src9_8.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_8.png` | 5 | `10:4 11:1` |
| `src9_9` | `build/inference assert/src9_9.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_9.png` | 4 | `9:2 10:2` |
| `src9_10` | `build/inference assert/src9_10.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_10.png` | 4 | `9:2 10:1 11:1` |
| `src9_11` | `build/inference assert/src9_11.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_11.png` | 3 | `9:1 10:1 11:1` |
| `src9_12` | `build/inference assert/src9_12.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_12.png` | 3 | `10:3` |
| `src9_13` | `build/inference assert/src9_13.png build/assert_matrix/model_9.json build/assert_matrix/model_10.json build/assert_matrix/model_11.json --metric ignore-local-polarity --min-score 0.9 --output build/assert_matrix/result_src9_13.png` | 2 | `9:2` |

所有命令均从仓库根目录执行。训练日志可用 shell 重定向单独保存，例如
`build/train ... > build/assert_matrix/train_1.log 2>&1`；推理日志同理。输出目录中的
`model_*.json`、`pyramid_*.png`、`result_*.png` 和推理自动生成的同名 `result_*.json`
不会覆盖 `assert/` 原图。CLI 矩阵是项目当前唯一的回归基线；若修改算法或参数，
应重新执行表中训练和推理命令并核对期望结果数量及 `template_id` 分布。

## 许可证

[MIT License](LICENSE)
