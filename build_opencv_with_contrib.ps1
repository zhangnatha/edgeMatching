[CmdletBinding()]
param(
    [ValidateSet('MinGW', 'MSVC')][string]$Toolchain = 'MinGW',
    [ValidateRange(1, 256)][int]$Jobs = [Environment]::ProcessorCount,
    [string]$InstallDir,
    [string]$Generator = 'Visual Studio 17 2022'
)

$ErrorActionPreference = 'Stop'
# Resolve defaults after parameter binding (Windows PowerShell compatibility).
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PSBoundParameters.ContainsKey('InstallDir')) { $InstallDir = Join-Path $scriptDir '3rdparty\opencv' }

$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$version = '4.7.0'
function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}
$tools = @('cmake.exe', 'curl.exe', 'tar.exe')
if ($Toolchain -eq 'MinGW') { $tools += @('gcc.exe', 'g++.exe', 'mingw32-make.exe') }
else { $tools += 'cl.exe' }
foreach ($tool in $tools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing required tool on PATH: $tool" }
}
if ($Toolchain -eq 'MinGW') {
    $target = & g++.exe -dumpmachine
    if ($LASTEXITCODE -ne 0 -or $target -notmatch '^x86_64-.*mingw') { throw 'Use an x64 MinGW-w64 compiler.' }
    if (Get-Command sh.exe -ErrorAction SilentlyContinue) { throw 'Remove Git/MSYS usr/bin from PATH and use native Windows PowerShell.' }
    $Generator = 'MinGW Makefiles'
} elseif ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
    throw 'Use an x64 Visual Studio developer shell.'
}

# Only this invocation owns the work directory; never remove a user-supplied directory.
$workDir = Join-Path $scriptDir ("edgeMatching-opencv-{0}" -f [guid]::NewGuid())
if ($InstallDir -eq $workDir -or $InstallDir.StartsWith($workDir + [IO.Path]::DirectorySeparatorChar)) {
    throw 'InstallDir must be outside the temporary build directory.'
}
New-Item -ItemType Directory -Path $workDir | Out-Null
$oldTmp = $env:TMP
$oldTemp = $env:TEMP
try {
    $env:TMP = Join-Path $workDir 'tmp'
    $env:TEMP = $env:TMP
    New-Item -ItemType Directory -Path $env:TMP | Out-Null
    foreach ($repo in 'opencv', 'opencv_contrib') {
        $archive = Join-Path $workDir "$repo.zip"
        Invoke-Checked 'curl.exe' @('-fL', '--connect-timeout', '30', '--retry', '3', '-o', $archive,
            "https://github.com/opencv/$repo/archive/refs/tags/$version.zip")
        Invoke-Checked 'tar.exe' @('-xf', $archive, '-C', $workDir)
    }
    $buildDir = Join-Path $workDir 'build'
    $args = @('-S', (Join-Path $workDir "opencv-$version"), '-B', $buildDir, '-G', $Generator,
        '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_INSTALL_PREFIX=$InstallDir",
        "-DOPENCV_EXTRA_MODULES_PATH=$(Join-Path $workDir "opencv_contrib-$version\modules")",
        '-DBUILD_LIST=core,imgproc,highgui,imgcodecs,calib3d,ximgproc',
        '-DBUILD_SHARED_LIBS=ON', '-DBUILD_TESTS=OFF', '-DBUILD_PERF_TESTS=OFF',
        '-DBUILD_EXAMPLES=OFF', '-DBUILD_opencv_apps=OFF', '-DBUILD_JAVA=OFF',
        '-DBUILD_opencv_python2=OFF', '-DBUILD_opencv_python3=OFF', '-DWITH_QT=OFF',
        '-DWITH_FFMPEG=OFF', '-DWITH_MSMF=OFF', '-DWITH_IPP=OFF')
    # OpenCV 4.7's pthread backend can reference pthread_self without a declaration
    # with some MinGW thread models. Select OpenMP instead; if unavailable OpenCV
    # falls back to sequential parallel_for while the application's OpenMP is independent.
    if ($Toolchain -eq 'MinGW') {
        $args += @('-DWITH_PTHREADS_PF=OFF', '-DWITH_OPENMP=ON', '-DWITH_TBB=OFF', '-DWITH_HPX=OFF')
    }
    if ($Generator -like 'Visual Studio *') { $args += @('-A', 'x64') }
    Invoke-Checked 'cmake.exe' $args
    Invoke-Checked 'cmake.exe' @('--build', $buildDir, '--config', 'Release', '--parallel', "$Jobs")
    Invoke-Checked 'cmake.exe' @('--install', $buildDir, '--config', 'Release')
    $config = Get-ChildItem $InstallDir -Recurse -Filter 'OpenCVConfig.cmake' | Select-Object -First 1
    $dlls = @(Get-ChildItem $InstallDir -Recurse -Filter '*.dll')
    if (-not $config -or -not $dlls.Count) { throw 'OpenCV installation is incomplete.' }
} catch {
    Write-Warning "OpenCV build failed. Files retained at $workDir"
    throw
} finally {
    $env:TMP = $oldTmp
    $env:TEMP = $oldTemp
}
Remove-Item -LiteralPath $workDir -Recurse -Force
Write-Host "OpenCV $version installed at $InstallDir; downloads and temporary build files removed."
