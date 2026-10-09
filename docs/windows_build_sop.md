# Windows 构建、验证与打包 SOP

本流程使用原生 Windows PowerShell，不使用 Git Bash。Qt、OpenCV、项目必须使用同一套 x64 MinGW-w64 编译器，不能混入 MSVC 二进制。Windows 实机测试尚未执行，以下命令需要在目标机器验收。构建目录建议使用短路径，例如 C:\src\edgeMatching。

## 1. 获取源码与准备环境

除 Git、CMake（至少 3.16）、MinGW-w64 外，源码构建 Qt 还需要 Perl、Python；样例验证需要 Python 3.6+。Windows 的 tar.exe、curl.exe 也必须可用。Perl 安装附带的其他 GCC 不要排在你的 MinGW 前面。请确认这套 GCC 提供 std::thread 支持。

```powershell
Set-Location C:\src
git clone https://github.com/zhangnatha/edgeMatching.git
Set-Location edgeMatching
# 改成实际 MinGW 目录；只添加 Git 的 cmd 目录，不添加 Git usr/bin。
$env:PATH = "C:\mingw64\bin;$env:PATH"
g++ --version
g++ -dumpmachine  # 应为 x86_64-...mingw...
mingw32-make --version
cmake --version
python --version
perl --version
where.exe g++.exe
where.exe sh.exe  # 应找不到；如找到，移除对应 Git/MSYS usr/bin PATH 项。
```

不要复用 MSVC 的构建目录或依赖安装目录。已有 MSVC 依赖时，下面的安装目录应另取名字，并相应修改后续命令。

## 2. 编译 OpenCV 4.7.0 与 contrib

以下按项目所需模块构建：core、imgproc、highgui、imgcodecs、calib3d，另外启用 contrib 的 ximgproc。需要其他 contrib 模块时扩展 BUILD_LIST。所有源文件、安装结果和临时目录均放在仓库内。

```powershell
$root = $PWD.Path
New-Item -ItemType Directory -Force 3rdparty\build_cache\tmp | Out-Null
$env:TEMP = "$root\3rdparty\build_cache\tmp"
$env:TMP = $env:TEMP
# curl 返回非零退出码时停止，不要继续解压。
curl.exe -fL --retry 3 -o 3rdparty\build_cache\opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
if ($LASTEXITCODE) { throw 'OpenCV download failed' }
curl.exe -fL --retry 3 -o 3rdparty\build_cache\contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
if ($LASTEXITCODE) { throw 'contrib download failed' }
Expand-Archive 3rdparty\build_cache\opencv.zip 3rdparty\build_cache -Force
Expand-Archive 3rdparty\build_cache\contrib.zip 3rdparty\build_cache -Force
cmake -S 3rdparty/build_cache/opencv-4.7.0 -B build-opencv-mingw -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/3rdparty/opencv" `
  -DOPENCV_EXTRA_MODULES_PATH="$root/3rdparty/build_cache/opencv_contrib-4.7.0/modules" `
  -DBUILD_LIST=core,imgproc,highgui,imgcodecs,calib3d,ximgproc `
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF `
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DWITH_QT=OFF `
  -DWITH_FFMPEG=OFF -DWITH_MSMF=OFF -DWITH_IPP=OFF
if ($LASTEXITCODE) { throw 'OpenCV configure failed' }
cmake --build build-opencv-mingw --parallel 8
if ($LASTEXITCODE) { throw 'OpenCV build failed' }
cmake --install build-opencv-mingw
if ($LASTEXITCODE) { throw 'OpenCV install failed' }
```

## 3. 编译本地 Qt、项目与验证

Qt 脚本默认仍为 MSVC，因此 MinGW 必须显式指定参数。缓存为 UI/build_cache，安装为 3rdparty/qt5，无需系统 Qt。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File UI/build_qt5.ps1 -Toolchain MinGW -Jobs 8
if ($LASTEXITCODE) { throw 'Qt build failed' }
$root = $PWD.Path
$opencvConfig = Get-ChildItem "$root\3rdparty\opencv" -Recurse -Filter OpenCVConfig.cmake | Select-Object -First 1
if (-not $opencvConfig) { throw 'OpenCVConfig.cmake not found' }
cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
  "-DOpenCV_DIR=$($opencvConfig.Directory.FullName)" `
  "-DQt5_DIR=$root/3rdparty/qt5/lib/cmake/Qt5" -DBUILD_QT_CLIENT=ON
if ($LASTEXITCODE) { throw 'Project configure failed' }
cmake --build build-mingw --parallel 8
if ($LASTEXITCODE) { throw 'Project build failed' }
$env:PATH = "$root\3rdparty\opencv\bin;$root\3rdparty\qt5\bin;$env:PATH"
Push-Location build-mingw
ctest --output-on-failure
$testExit = $LASTEXITCODE
Pop-Location
if ($testExit) { throw 'Tests failed' }
.\build-mingw\train.exe assert/m8.bmp --id 8 --output build-mingw/model_8.json
.\build-mingw\inference.exe assert/src8.bmp build-mingw/model_8.json --min-score 0.95 --min-visible-ratio 0.5 --subpixel --output build-mingw/src8.result.png
.\3rdparty\qt5\bin\windeployqt.exe --release .\build-mingw\UI\shape_match_qt.exe
.\build-mingw\UI\shape_match_qt.exe
```

CTest 应包含 assert_matrix 与 qt_client_startup，样例矩阵验证 11 次训练和 26 次推理；src8 应识别 7 个实例。MinGW 是单配置生成器，输出不在 Release 子目录。

## 4. 打包与安装

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_release.ps1 `
  -Generator "MinGW Makefiles" -BuildDir build-mingw-release `
  -OpenCVDir 3rdparty/opencv -QtDir 3rdparty/qt5
if ($LASTEXITCODE) { throw 'Packaging failed' }
.\dist\edgeMatching-windows\bin\shape_match_qt.exe --smoke-test
```

输出 dist/edgeMatching-windows 文件夹和同名 ZIP。脚本收集项目 DLL、OpenCV DLL、MinGW 运行库，并调用本地 Qt 的 windeployqt 收集 Qt DLL 和平台插件。

这是便携安装包，不生成 setup.exe/MSI。复制整个文件夹到目标机器或解压 ZIP，即可从 bin/shape_match_qt.exe 启动；CLI 在 bin/train.exe、bin/inference.exe。不要只复制 exe。无需在目标机器安装 Qt、OpenCV、GCC 或 CMake。

发布前分别在 Windows 10、11 干净机器上检查 GUI、训练与推理；测试时 PATH 不包含开发机 Qt、OpenCV 或 MinGW 路径，以发现漏打包 DLL。ZIP 不携带 assert/*.bmp；验收样例需另从仓库准备。如果需要向导安装器，可用 Inno Setup/NSIS 将整个发布目录作为 payload，入口为 bin/shape_match_qt.exe。

参考：[CMake MinGW Makefiles](https://cmake.org/cmake/help/latest/generator/MinGW%20Makefiles.html)、[Qt 5 构建说明](https://wiki.qt.io/Building_Qt_5_from_Git)。

## 5. 可选 MSVC 构建流程

若改用 Visual Studio 2019/2022，应在 x64 开发者 PowerShell 中执行，使用 MSVC 版 OpenCV，并用独立目录保存依赖与构建结果。以下假定官方 OpenCV Windows 包已解压到 `3rdparty/opencv-msvc`，其下包含 `build/OpenCVConfig.cmake`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File UI/build_qt5.ps1 `
  -Toolchain MSVC -Jobs 8 -InstallDir 3rdparty/qt5-msvc
if ($LASTEXITCODE) { throw 'Qt MSVC build failed' }
$root = $PWD.Path
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 `
  "-DOpenCV_DIR=$root/3rdparty/opencv-msvc/build" `
  "-DQt5_DIR=$root/3rdparty/qt5-msvc/lib/cmake/Qt5" -DBUILD_QT_CLIENT=ON
if ($LASTEXITCODE) { throw 'MSVC configure failed' }
cmake --build build-msvc --config Release --parallel 8
if ($LASTEXITCODE) { throw 'MSVC build failed' }
$env:PATH = "$root\3rdparty\qt5-msvc\bin;$root\3rdparty\opencv-msvc\build\x64\vc16\bin;$env:PATH"
Push-Location build-msvc
ctest -C Release --output-on-failure
$testExit = $LASTEXITCODE
Pop-Location
if ($testExit) { throw 'MSVC tests failed' }
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_release.ps1 `
  -Generator "Visual Studio 17 2022" -BuildDir build-msvc-release `
  -OpenCVDir 3rdparty/opencv-msvc -QtDir 3rdparty/qt5-msvc `
  -OutputDir dist/edgeMatching-windows-msvc
if ($LASTEXITCODE) { throw 'MSVC packaging failed' }
```

Visual Studio 2019 使用生成器 `Visual Studio 16 2019`；OpenCV 的 DLL 目录应根据实际包布局调整。MSVC 的 GUI 输出为 `build-msvc/UI/Release/shape_match_qt.exe`。Qt 有 `jom.exe` 时并行构建，否则使用串行 `nmake`。

仅编译 CLI 可关闭 `BUILD_QT_CLIENT`；仅打包 CLI 在发布命令加 `-NoQt`，无需 Qt 构建步骤。

持续集成配置见 [build.yml](../.github/workflows/build.yml)，其中 Windows Server 2022 任务使用 MSVC。该任务不能代替 Windows 10/11 实机验收，也不覆盖 MinGW 构建。
