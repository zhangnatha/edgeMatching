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

function Ensure-PerlTool([string]$CacheDirectory, [string]$ToolchainName) {
    if (Get-Command perl.exe -ErrorAction SilentlyContinue) {
        return
    }
    foreach ($cand in @('C:\Strawberry\perl\bin', 'D:\Strawberry\perl\bin')) {
        if (Test-Path (Join-Path $cand 'perl.exe')) {
            $env:PATH = "$cand;$env:PATH"
            Write-Host "Found Strawberry Perl at $cand; added to PATH."
            return
        }
    }
    $gitCmd = Get-Command git.exe -ErrorAction SilentlyContinue
    $gitPerlCandidates = @()
    if ($gitCmd) {
        $gitRoot = Split-Path -Parent (Split-Path -Parent $gitCmd.Source)
        $gitPerlCandidates += (Join-Path $gitRoot 'usr\bin\perl.exe')
    }
    $gitPerlCandidates += @(
        'D:\Program Files\Git\usr\bin\perl.exe',
        'C:\Program Files\Git\usr\bin\perl.exe',
        'C:\Program Files (x86)\Git\usr\bin\perl.exe'
    )
    $foundGitPerl = $null
    foreach ($cand in $gitPerlCandidates) {
        if (Test-Path -LiteralPath $cand) {
            $foundGitPerl = [IO.Path]::GetFullPath($cand)
            break
        }
    }
    if ($foundGitPerl) {
        $toolsDir = Join-Path $CacheDirectory 'tools\bin'
        New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
        $shimExe = Join-Path $toolsDir 'perl.exe'
        if (-not (Test-Path -LiteralPath $shimExe)) {
            $shimSrc = Join-Path $CacheDirectory 'tools\perl_shim.c'
            $escapedTarget = $foundGitPerl.Replace('\', '\\')
            $cCode = @"
#include <windows.h>
#include <stdio.h>

int main() {
    LPWSTR cmdLine = GetCommandLineW();
    const wchar_t* target = L"$escapedTarget";
    const wchar_t* p = cmdLine;
    if (*p == L'"') { p++; while (*p && *p != L'"') p++; if (*p == L'"') p++; }
    else { while (*p && *p != L' ') p++; }
    while (*p == L' ') p++;
    wchar_t newCmd[32768];
    _snwprintf(newCmd, 32768, L"\"%s\" %s", target, p);
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    if (!CreateProcessW(NULL, newCmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) return 1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)exitCode;
}
"@
            Set-Content -LiteralPath $shimSrc -Value $cCode -Encoding UTF8
            if ($ToolchainName -eq 'MinGW' -or (Get-Command gcc.exe -ErrorAction SilentlyContinue)) {
                Invoke-Checked 'gcc.exe' @('-O2', $shimSrc, '-o', $shimExe)
            } else {
                Invoke-Checked 'cl.exe' @('/O2', "/Fe:$shimExe", $shimSrc)
            }
            Remove-Item -LiteralPath $shimSrc -Force
        }
        $env:PATH = "$toolsDir;$env:PATH"
        Write-Host "Configured isolated Perl forwarder pointing to $foundGitPerl (avoids sh.exe on PATH)."
        return
    }
    throw 'perl.exe is required to configure and build Qt. Please install Strawberry Perl or Git for Windows.'
}

$oldPath = $env:PATH
Ensure-PerlTool $CacheDir $Toolchain

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
    $fsEngine = Join-Path $sourceDir 'qtbase\src\corelib\io\qfilesystemengine_win.cpp'
    if (Test-Path $fsEngine) {
        $content = Get-Content -LiteralPath $fsEngine -Raw
        $oldDef = '#if defined(Q_CC_MINGW) && WINVER < 0x0602 //  Windows 8 onwards'
        $newDef = '#if defined(Q_CC_MINGW) && WINVER < 0x0602 && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602) //  Windows 8 onwards'
        if ($content.Contains($oldDef)) {
            $content = $content.Replace($oldDef, $newDef)
            Set-Content -LiteralPath $fsEngine -Value $content -Encoding UTF8
        }
    }
    $qCollGen = Join-Path $sourceDir 'qttools\src\assistant\qcollectiongenerator\main.c'
    if (Test-Path $qCollGen) {
        $content = Get-Content -LiteralPath $qCollGen -Raw
        $oldSpawn = '_spawnvp(_P_WAIT, newPath, argv)'
        $newSpawn = '_spawnvp(_P_WAIT, newPath, (const char * const *)argv)'
        if ($content.Contains($oldSpawn)) {
            $content = $content.Replace($oldSpawn, $newSpawn)
            Set-Content -LiteralPath $qCollGen -Value $content -Encoding UTF8
        }
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
    $env:PATH = $oldPath
    foreach ($name in $savedNativeEnv.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedNativeEnv[$name], 'Process')
    }
}
