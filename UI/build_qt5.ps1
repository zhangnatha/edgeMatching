[CmdletBinding()]
param(
    [ValidateSet('MSVC', 'MinGW')][string]$Toolchain = 'MSVC',
    [ValidateRange(1, 256)][int]$Jobs = [Environment]::ProcessorCount,
    [string]$CacheDir,
    [string]$InstallDir,
    [string]$Url = 'https://download.qt.io/archive/qt/5.15/5.15.16/single/qt-everywhere-opensource-src-5.15.16.tar.xz'
)

# MSVC requires a developer shell; MinGW requires a native Windows PowerShell.
$ErrorActionPreference = 'Stop'
# Resolve defaults after parameter binding (Windows PowerShell compatibility).
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PSBoundParameters.ContainsKey('CacheDir')) { $CacheDir = Join-Path $scriptDir 'build_cache' }
if (-not $PSBoundParameters.ContainsKey('InstallDir')) { $InstallDir = Join-Path $scriptDir '..\3rdparty\qt5' }

$version = '5.15.16'
$sha256 = 'efa99827027782974356aceff8a52bd3d2a8a93a54dd0db4cca41b5e35f1041c'
$CacheDir = [IO.Path]::GetFullPath($CacheDir)
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$archive = Join-Path $CacheDir "qt-everywhere-opensource-src-$version.tar.xz"
$sourceDir = Join-Path $CacheDir "src-$version"
$buildDir = Join-Path $CacheDir "build-$version-$($Toolchain.ToLowerInvariant())"

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

$requiredTools = @('perl.exe', 'python.exe', 'curl.exe')
if ($Toolchain -eq 'MinGW') { $requiredTools += @('gcc.exe', 'g++.exe', 'mingw32-make.exe') }
else { $requiredTools += @('cl.exe', 'nmake.exe') }
foreach ($tool in $requiredTools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing required tool on PATH: $tool" }
}
$platform = 'win32-msvc'
if ($Toolchain -eq 'MinGW') {
    $platform = 'win32-g++'
    $target = & g++.exe -dumpmachine
    if ($LASTEXITCODE -ne 0 -or $target -notmatch '^x86_64-.*mingw') {
        throw 'Use an x64 MinGW-w64 compiler for Qt, OpenCV and the application.'
    }
    if (Get-Command sh.exe -ErrorAction SilentlyContinue) {
        throw 'Remove Git usr/bin or MSYS usr/bin from PATH so sh.exe is unavailable. Run in native Windows PowerShell.'
    }
} elseif ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
    throw 'Qt and OpenCV must both be built for x64.'
}
function Remove-QtDownloads {
    foreach ($file in @($archive, "$archive.part")) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
    }
}
$qmake = Join-Path $InstallDir 'bin\qmake.exe'
if ((Test-Path $qmake) -and (Test-Path (Join-Path $InstallDir 'bin\lrelease.exe')) -and
    (Test-Path (Join-Path $InstallDir 'bin\windeployqt.exe')) -and
    (Test-Path (Join-Path $InstallDir 'lib\cmake\Qt5Widgets\Qt5WidgetsConfig.cmake')) -and
    (Test-Path (Join-Path $InstallDir 'lib\cmake\Qt5Concurrent\Qt5ConcurrentConfig.cmake')) -and
    (Test-Path (Join-Path $InstallDir 'plugins\platforms\qwindows.dll'))) {
    $installedSpec = & $qmake -query QMAKE_XSPEC
    if ($LASTEXITCODE -ne 0 -or $installedSpec -ne $platform) {
        throw "Existing Qt uses $installedSpec; requested $platform. Choose a separate -InstallDir."
    }
    $installedVersion = & $qmake -query QT_VERSION
    if ($LASTEXITCODE -eq 0 -and $installedVersion -eq $version) {
        Remove-QtDownloads
        Write-Host "Qt $version already installed at $InstallDir"
        return
    }
}

New-Item -ItemType Directory -Force -Path $CacheDir, $buildDir | Out-Null
$oldTmp = $env:TMP
$oldTemp = $env:TEMP
$nativeEnvNames = @('CC', 'CXX', 'CPP', 'CROSS_COMPILE', 'MAKEFLAGS', 'MFLAGS',
    'MAKEOVERRIDES', 'SHELL', 'MSYSTEM', 'QMAKESPEC', 'XQMAKESPEC', 'QMAKEPATH', 'QMAKEFEATURES')
$savedNativeEnv = @{}
try {
    if ($Toolchain -eq 'MinGW') {
        foreach ($name in $nativeEnvNames) {
            $savedNativeEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        # Apply to configure.bat's bootstrap make as well as later recursive builds.
        # Qt's Makefile.unix.mingw uses SH=0 to select native del/rmdir commands.
        $env:MAKEFLAGS = 'CC=gcc CXX=g++ SHELL=cmd.exe SH=0'
        Write-Host "Using native MinGW compiler: $((Get-Command 'g++.exe').Source)"
    }
    $env:TMP = Join-Path $CacheDir 'tmp'
    $env:TEMP = $env:TMP
    New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
    if (-not (Test-Path $archive)) {
        $downloaded = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            & curl.exe -fL --connect-timeout 30 --retry 3 -C - -o "$archive.part" $Url
            if ($LASTEXITCODE -eq 0) { $downloaded = $true; break }
            Write-Warning "Download attempt $attempt failed; partial download retained."
        }
        if (-not $downloaded) { throw 'Qt download failed' }
        Move-Item "$archive.part" $archive
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha256) {
        throw "Qt archive SHA256 mismatch: $archive"
    }
    $extractMarker = Join-Path $sourceDir '.shape-match-extracted'
    if (-not (Test-Path $extractMarker) -or -not (Test-Path (Join-Path $sourceDir 'configure.bat'))) {
        New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
        # Python handles xz directly; Windows tar may depend on a missing xz.exe.
        $extractCode = @'
import os, sys, tarfile
import ntpath

def extraction_path(path, windows):
    if not windows:
        return os.path.abspath(path)
    path = ntpath.abspath(path)
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path

with tarfile.open(sys.argv[1], "r:xz") as archive:
    members = []
    for member in archive.getmembers():
        parts = member.name.split("/", 1)
        if len(parts) == 2 and parts[1]:
            member.name = parts[1]
            if os.name == "nt":
                member.name = member.name.replace("/", "\\")
                if member.issym() or member.islnk():
                    member.linkname = member.linkname.replace("/", "\\")
            members.append(member)
    archive.extractall(extraction_path(sys.argv[2], os.name == "nt"), members=members)
'@
        $extractScript = Join-Path $CacheDir 'extract-qt-source.py'
        # Windows PowerShell 5.1 can strip quotes from native -c arguments.
        Set-Content -LiteralPath $extractScript -Value $extractCode -Encoding ASCII
        Invoke-Checked 'python.exe' @($extractScript, $archive, $sourceDir)
        Remove-Item -LiteralPath $extractScript -Force
        Set-Content -Path $extractMarker -Value $sha256 -Encoding ASCII
    }
    Push-Location $buildDir
    try {
        $configFile = Join-Path $buildDir '.shape-match-qt-config'
        $configSignature = "$sourceDir|$InstallDir|widgets-desktop-opengl-no-egl-angle-v1"
        if (-not (Test-Path 'Makefile') -or -not (Test-Path $configFile) -or
            (Get-Content $configFile -Raw).Trim() -ne $configSignature) {
            $configureArgs = @('-prefix', $InstallDir, '-opensource', '-confirm-license',
                '-release', '-platform', $platform, '-opengl', 'desktop', '-no-egl', '-no-angle', '-nomake', 'examples', '-nomake', 'tests', '-no-feature-qdoc')
            foreach ($module in @('qt3d', 'qtactiveqt', 'qtandroidextras', 'qtcharts',
                'qtconnectivity', 'qtdatavis3d', 'qtdeclarative', 'qtgamepad', 'qtlocation',
                'qtlottie', 'qtmultimedia', 'qtnetworkauth', 'qtpurchasing', 'qtquick3d',
                'qtremoteobjects', 'qtscript', 'qtscxml', 'qtsensors', 'qtserialbus',
                'qtserialport', 'qtspeech', 'qttranslations', 'qtwebengine', 'qtvirtualkeyboard',
                'qtwebchannel', 'qtwebglplugin', 'qtwebsockets', 'qtwebview', 'qtx11extras')) {
                $configureArgs += @('-skip', $module)
            }
            Invoke-Checked (Join-Path $sourceDir 'configure.bat') $configureArgs
            Set-Content -Path $configFile -Value $configSignature -Encoding UTF8
        }
        if ($Toolchain -eq 'MinGW') {
            Invoke-Checked 'mingw32-make.exe' @('-j', "$Jobs")
            Invoke-Checked 'mingw32-make.exe' @('install')
        } elseif (Get-Command 'jom.exe' -ErrorAction SilentlyContinue) {
            Invoke-Checked 'jom.exe' @('-j', "$Jobs")
            Invoke-Checked 'jom.exe' @('install')
        } else {
            Write-Host 'jom.exe is unavailable; using serial nmake.'
            Invoke-Checked 'nmake.exe' @('/NOLOGO')
            Invoke-Checked 'nmake.exe' @('/NOLOGO', 'install')
        }
    } finally { Pop-Location }
    foreach ($required in @('bin\qmake.exe', 'bin\lrelease.exe', 'bin\windeployqt.exe',
        'plugins\platforms\qwindows.dll', 'lib\cmake\Qt5Widgets\Qt5WidgetsConfig.cmake',
        'lib\cmake\Qt5Concurrent\Qt5ConcurrentConfig.cmake')) {
        if (-not (Test-Path (Join-Path $InstallDir $required))) { throw "Qt installation incomplete: $required" }
    }
    Remove-QtDownloads
    Write-Host "Qt $version installed at $InstallDir; downloaded archives removed."
} catch {
    Write-Warning "Qt build failed. Cache retained at $CacheDir"
    throw
} finally {
    $env:TMP = $oldTmp
    $env:TEMP = $oldTemp
    foreach ($name in $savedNativeEnv.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedNativeEnv[$name], 'Process')
    }
}
