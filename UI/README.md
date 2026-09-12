# Shape Match Qt5 客户端

这是核心库的 Qt5 Widgets 客户端。所有客户端源码、头文件和脚本均位于 `UI/`；顶层 CMake 默认构建客户端，无 Qt 环境时可关闭。

## 依赖与构建

需要 Qt5 Widgets/Concurrent、OpenCV，以及仓库根目录先构建出的 `MakeTemplate` 和 `FindTemplate`。构建方式：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/UI/shape_match_qt
```

无 Qt 环境时使用 `-DBUILD_QT_CLIENT=OFF`；仓库根目录构建会自动优先发现 `3rdparty/qt5`。

客户端生成的模型（JSON/BIN）、推理结果图、结果 JSON 以及模型金字塔预览默认保存到
`QCoreApplication::applicationDirPath()`，即 `shape_match_qt` 可执行文件同级目录（通常为
`build/UI/`）。编辑框中的相对路径也会规范到该目录；用户明确选择的绝对路径会保留。

亚像素精修、AVX2 SIMD 和边缘特征后端均在运行时设置，无需重编译客户端。

## Qt5 安装脚本

`build_qt5.sh` 下载官方 Qt 5.15.16 源码，校验固定 SHA256 后，将最小模块安装到仓库 `3rdparty/qt5`。源码、压缩包和构建目录默认在 `/tmp/shape-match-qt5`；脚本不会自动执行下载。

```bash
JOBS=8 UI/build_qt5.sh
# 或：UI/build_qt5.sh --jobs 8 --cache /tmp/qt-cache
```

可用环境变量：`QT_SHA256`（校验覆盖）、`QT_CACHE_DIR`、`QT_SRC_DIR`、`QT_BUILD_DIR`、`QT_ARCHIVE`、`QT_INSTALL_DIR`。重复运行时若检测到 `<install>/bin/qmake` 会直接退出。

脚本不使用 `sudo`，会在缺少依赖时直接报错；请由用户或管理员预先安装构建依赖。Linux/XCB 常见依赖包括 `gcc`、`g++`、`make`、`perl`、`python3`、`awk`、`gperf`、`bison`、`flex`、`libfontconfig1-dev`、`libdbus-1-dev`、`libfreetype6-dev`、`libicu-dev`、`libglib2.0-dev`、`libx11-dev`、`libx11-xcb-dev`、`libxcb1-dev`、`libxcb-*`、`libxext-dev`、`libxfixes-dev`、`libxi-dev`、`libxrender-dev`、`libxkbcommon-dev`、`libxkbcommon-x11-dev`。不同发行版的包名可能不同。

根目录配置会自动优先发现 `3rdparty/qt5/lib/cmake/Qt5/Qt5Config.cmake`。若该目录不存在，`BUILD_QT_CLIENT=ON` 会明确报错提示安装/指定 Qt5，而不会跳过客户端目标。

## 操作与参数映射

左侧“训练模板”保留原始模板图像色彩，并以 HALCON 风格叠加连续的 canonical 亚像素边缘轮廓。训练完成后右侧“训练参数”中的“显示金字塔层”会列出实际生成的 L0..Ln（标签包含层尺寸和特征数），默认显示 L0；切换层时使用与核心相同的整数半尺寸 `pyrDown` 序列，并从对应 `templates[level]` 的 canonical 特征恢复该层轮廓。空层保留标注并不绘制轮廓。预览会先把特征中心量化到像素 mask，再用 `findContours` 恢复连通组件及孔洞拓扑，最后将轮廓节点映射回原始亚像素坐标，通过 `ImageView` 独立的 QPainter 矢量前景层以约 1px 的抗锯齿绿色折线绘制；绿色轮廓不会烧录进原图或 QPixmap。高倍缩放时仅图像层显示最近邻像素格，矢量轮廓保持平滑。内部模型金字塔仍按训练参数生成并用于匹配。 “推理结果”同样保留原始输入图像，按每个结果的 `template_id` 选择对应模型，在 L0 canonical 特征上按最终 `(x,y,angle,scale)` 生成绿色连续轮廓，并以青色矢量旋转框和姿态箭头叠加；所有覆盖均在 `ImageView::drawForeground` 中绘制，不会烧录到 `cv::Mat`，因此缩放时保持 HALCON 式干净线条。两个图像区域都支持以鼠标位置为锚点的滚轮缩放、拖拽平移、“适应窗口”和“1:1”。

图像视图在每个源像素放大到约 8 个屏幕像素后显示灰色像素网格，网格只绘制在图像矩形内并按可视区域裁剪；此时关闭平滑插值以保持像素块清晰，缩回低倍率后自动恢复平滑缩放。

右侧参数区域分为“训练参数”和“推理参数”两个标签页，每页独立滚动；下方为公共进度条和状态日志。左侧图像标签页与右侧参数标签页联动，切换工作流时自动显示对应内容。

训练参数直接映射 `CreateTemplate::createTemplate`：Template ID、金字塔层数（-1 自动或 0..7）、角度起止/步长、Otsu、最小/最大对比度及 JSON/BIN 输出。边缘算法提供 Canny 像素级、Canny + 抛物线亚像素、Devernay 亚像素三项；旧模型中的 `edge_method=0/1` 仍分别按 Current/Devernay 加载。训练成功后“显示金字塔层”按实际模型层数启用，默认 L0，可切换查看每层缩小图和该层 `ShapeInfo` 特征。

推理参数直接映射 `SearchTemplate` 多模板/ROI 重载与 `ScaleSearchCfg`：输入图像、多个 JSON/BIN 模型、ROI x/y/w/h、角度范围、最小得分、最大匹配数、最大重叠、搜索层数、贪婪度、Y 排序、尺度 min/max/step、最小可见比例、最小对比度、metric、subpixel 和 SIMD 请求。训练参数另有 CURRENT/Devernay 边缘算法下拉框。训练及推理均在 `QtConcurrent` 后台任务中运行，结果回到 GUI 线程后才更新模型和图像。

注意：现有公开搜索 API 的参数名为 `angle_extent`，但核心实现和 CLI 约定它接收绝对终止角度；客户端因此传入“起始角度”和“终止角度”本身，而不是角度差值。

## Language / 语言

Use the `Language` menu in the title bar to switch between 中文 and English at
runtime. The selection is stored with `QSettings`, so the next launch keeps the
last choice. All visible labels, tabs, buttons, parameter names, tooltips,
status messages and error dialogs use the same translation catalog. The English
catalog is `shape_match_en.ts`; the top-level CMake build invokes `lrelease` and
copies `shape_match_en.qm` beside `build/UI/shape_match_qt`.

使用标题栏中的“语言 / Language”菜单可实时切换中文和 English。语言偏好由
`QSettings` 保存，下一次启动会继续使用上次选择。所有可见标签、标签页、按钮、
参数项、提示、状态日志和错误对话框统一使用翻译目录；顶层 CMake 一次构建会调用
`lrelease` 并将 `shape_match_en.qm` 复制到 `build/UI/shape_match_qt` 同级目录。
