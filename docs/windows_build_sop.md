# Windows 构建与打包 SOP

适用于 Windows 10/11，使用原生 PowerShell 和同一套 x64 MinGW-w64。
提前安装 Git、CMake 3.16+、MinGW-w64、Perl、Python 3.6+，确认 `tar.exe`、
`curl.exe` 可用。MinGW 的 `bin` 放在 PATH 前面，Git/MSYS 的 `usr/bin` 不加入 PATH。
以下命令在仓库根目录执行，每一步成功后再继续。

## 1. 安装第三方库

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build_opencv_with_contrib.ps1 -Jobs 8
powershell -NoProfile -ExecutionPolicy Bypass -File UI/build_qt5.ps1 -Toolchain MinGW -Jobs 8
```

OpenCV、Qt 分别安装到 `3rdparty/opencv`、`3rdparty/qt5`，无需系统 Qt。
成功后自动清理下载压缩包；OpenCV 同时清理本次临时构建目录，Qt 保留源码和构建缓存。
失败时保留文件供排查。两个脚本均支持 `-InstallDir`。

## 2. 编译与验证

```powershell
$root = $PWD.Path
$ocv = Get-ChildItem 3rdparty/opencv -Recurse -Filter OpenCVConfig.cmake | Select-Object -First 1
cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
  "-DOpenCV_DIR=$($ocv.Directory.FullName)" `
  "-DQt5_DIR=$root/3rdparty/qt5/lib/cmake/Qt5" -DBUILD_QT_CLIENT=ON
cmake --build build-mingw --parallel 8
$env:PATH = "$root\3rdparty\opencv\bin;$root\3rdparty\qt5\bin;$env:PATH"
Push-Location build-mingw
ctest --output-on-failure
Pop-Location
```

CTest 验证 11 次训练、26 次推理及 Qt 启动。CLI 位于 `build-mingw/train.exe`、
`build-mingw/inference.exe`；样例参数见 [README](../README.md)。

## 3. 打包与运行

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_release.ps1 `
  -Generator "MinGW Makefiles" -BuildDir build-mingw-release
.\dist\edgeMatching-windows\bin\shape_match_qt.exe
```

生成 `dist/edgeMatching-windows` 和同名 ZIP。将整个目录复制或解压到目标机器，
从 `bin` 启动即可，无需安装 Qt、OpenCV 或编译工具。当前为便携包，不生成安装向导。
Windows 实机验收尚未完成，发布前应在 Windows 10/11 上验证。

使用 MSVC 时，在 x64 Visual Studio 开发者 PowerShell 中给两个依赖脚本传入
`-Toolchain MSVC`，项目和打包生成器改为 `Visual Studio 17 2022`，项目配置加
`-A x64`，编译及 CTest 加 `--config Release` / `-C Release`。不同工具链使用独立
依赖安装目录与构建目录，不混用二进制。
