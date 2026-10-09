[CmdletBinding()]
param(
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build-release'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\dist\edgeMatching-windows'),
    [string]$OpenCVDir = (Join-Path $PSScriptRoot '..\3rdparty\opencv'),
    [string]$QtDir = (Join-Path $PSScriptRoot '..\3rdparty\qt5'),
    [string]$Generator = 'Visual Studio 17 2022',
    [switch]$NoQt
)

$ErrorActionPreference = 'Stop'
$repoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$OpenCVDir = [IO.Path]::GetFullPath($OpenCVDir)
$QtDir = [IO.Path]::GetFullPath($QtDir)

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'Missing required command: cmake' }
if ($OutputDir -eq [IO.Path]::GetPathRoot($OutputDir) -or $OutputDir -eq $repoDir -or
    $OutputDir -eq $BuildDir -or $BuildDir.StartsWith($OutputDir + [IO.Path]::DirectorySeparatorChar) -or
    ($OutputDir.StartsWith($repoDir + [IO.Path]::DirectorySeparatorChar) -and
     -not $OutputDir.StartsWith((Join-Path $repoDir 'dist') + [IO.Path]::DirectorySeparatorChar) -and
     $OutputDir -ne (Join-Path $repoDir 'dist'))) {
    throw "Refusing unsafe output directory: $OutputDir"
}

# 根据本地依赖判断是否构建 Qt 客户端。
$qtConfig = Join-Path $QtDir 'lib\cmake\Qt5\Qt5Config.cmake'
$qtEnabled = -not $NoQt
if ($qtEnabled -and -not (Test-Path $qtConfig)) {
    throw 'Local Qt5 not found. Run UI\build_qt5.ps1, specify -QtDir, or use -NoQt.'
}
$qtFlag = if ($qtEnabled) { 'ON' } else { 'OFF' }
$cmakeArgs = @('-S', $repoDir, '-B', $BuildDir, '-G', $Generator, '-U', 'Qt5*',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DBUILD_QT_CLIENT=$qtFlag")
if ($Generator -like 'Visual Studio *') { $cmakeArgs += @('-A', 'x64') }
$opencvConfigDir = $null
foreach ($candidate in @($OpenCVDir, (Join-Path $OpenCVDir 'build'), (Join-Path $OpenCVDir 'lib\cmake\opencv4'))) {
    if (Test-Path (Join-Path $candidate 'OpenCVConfig.cmake')) { $opencvConfigDir = $candidate; break }
}
if (-not $opencvConfigDir) { throw "OpenCVConfig.cmake not found below $OpenCVDir" }
$cmakeArgs += "-DOpenCV_DIR=$opencvConfigDir"
if ($qtEnabled) { $cmakeArgs += @("-DCMAKE_PREFIX_PATH=$QtDir", "-DQt5_DIR=$(Split-Path $qtConfig)") }

Write-Host "Configuring release build (Qt client: $qtEnabled)..."
Invoke-Checked 'cmake' $cmakeArgs
Write-Host 'Building release binaries...'
Invoke-Checked 'cmake' @('--build', $BuildDir, '--config', 'Release', '--parallel')

$stage = Join-Path $repoDir ("edgeMatching-release-{0}" -f ([guid]::NewGuid()))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    Invoke-Checked 'cmake' @('--install', $BuildDir, '--config', 'Release', '--prefix', $stage)
    $binDir = Join-Path $stage 'bin'
    $libDir = Join-Path $stage 'lib'
    New-Item -ItemType Directory -Force -Path $libDir | Out-Null

    # 收集 MSVC 运行时所需的项目 DLL 和 OpenCV DLL。
    foreach ($name in 'MakeTemplate.dll', 'FindTemplate.dll') {
        $candidate = Join-Path $BuildDir "Release\$name"
        if (Test-Path $candidate) { Copy-Item $candidate $binDir -Force }
    }
    # 官方 Windows 包的 DLL 位于 build\x64\vc16\bin，源码安装通常位于 bin。
    $opencvDlls = @(Get-ChildItem $OpenCVDir -Recurse -File -Filter '*.dll' |
        Where-Object { $_.Directory.Name -eq 'bin' -and $_.FullName -notmatch '[\\/]x86[\\/]' })
    if (-not $opencvDlls.Count) { throw "OpenCV runtime DLLs not found below $OpenCVDir" }
    $opencvDlls | Copy-Item -Destination $binDir -Force

    # 复制 MSVC 的可再分发运行库，使 CLI 也能在未安装 Visual Studio 的机器上运行。
    if ($Generator -ne 'MinGW Makefiles' -and ($Generator -like 'Visual Studio *' -or $env:VCToolsRedistDir)) {
        $redistDir = $env:VCToolsRedistDir
        if (-not $redistDir) {
            $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (Test-Path $vswhere) {
                $vsInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
                if ($LASTEXITCODE -ne 0) { throw 'vswhere failed' }
                if ($vsInstall) {
                    $redistRoot = Join-Path $vsInstall 'VC\Redist\MSVC'
                    $latest = Get-ChildItem $redistRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
                    if ($latest) { $redistDir = $latest.FullName }
                }
            }
        }
        if (-not $redistDir) { throw 'MSVC redistributable directory not found. Use a Visual Studio developer shell.' }
        $crtDirs = @(Get-ChildItem (Join-Path $redistDir 'x64') -Directory -Filter 'Microsoft.VC*.CRT')
        if (-not $crtDirs.Count) { throw "MSVC x64 runtime DLLs not found in $redistDir" }
        foreach ($crt in $crtDirs) { Copy-Item (Join-Path $crt.FullName '*.dll') $binDir -Force }
        $openmpDirs = @(Get-ChildItem (Join-Path $redistDir 'x64') -Directory -Filter 'Microsoft.VC*.OpenMP')
        foreach ($openmp in $openmpDirs) { Copy-Item (Join-Path $openmp.FullName '*.dll') $binDir -Force }
    } elseif ($Generator -eq 'MinGW Makefiles') {
        $compiler = (Get-Command 'g++.exe' -ErrorAction Stop).Source
        foreach ($name in 'libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll', 'libgomp-1.dll') {
            $runtime = & $compiler "-print-file-name=$name"
            if ($LASTEXITCODE -ne 0) { throw "Cannot locate MinGW runtime: $name" }
            if (-not (Test-Path $runtime)) { $runtime = Join-Path (Split-Path $compiler) $name }
            if (Test-Path $runtime) { Copy-Item $runtime $binDir -Force }
        }
    }

    if ($qtEnabled) {
        $qtDeploy = Join-Path $QtDir 'bin\windeployqt.exe'
        if (-not (Test-Path $qtDeploy)) { throw "windeployqt not found: $qtDeploy" }
        $qtExe = Join-Path $binDir 'shape_match_qt.exe'
        if (-not (Test-Path $qtExe)) { throw "Qt executable not found: $qtExe" }
        Invoke-Checked $qtDeploy @('--release', '--no-translations', $qtExe)
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $stage 'docs') | Out-Null
    Copy-Item (Join-Path $repoDir 'README.md'), (Join-Path $repoDir 'LICENSE') $stage -Force
    Copy-Item (Join-Path $repoDir 'docs\template_matching_algorithm.md') (Join-Path $stage 'docs') -Force
    $assertMdDir = Join-Path $repoDir 'assert\.md'
    if (Test-Path $assertMdDir) {
        $stageAssertMd = Join-Path $stage 'assert\.md'
        New-Item -ItemType Directory -Force -Path $stageAssertMd | Out-Null
        Copy-Item (Join-Path $assertMdDir '*') $stageAssertMd -Recurse -Force
    }
    Set-Content -Path (Join-Path $stage 'RELEASE.txt') -Value @(
        'edgeMatching release package', "Built from: $repoDir", "Qt client: $qtEnabled"
    ) -Encoding UTF8

    if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path $OutputDir) | Out-Null
    Move-Item $stage $OutputDir
    $stage = $null
    Compress-Archive -Path (Join-Path $OutputDir '*') -DestinationPath ($OutputDir + '.zip') -Force
    Write-Host "Release package created at $OutputDir"
    Write-Host "ZIP archive created at $OutputDir.zip"
}
finally {
    if ($stage -and (Test-Path $stage)) { Remove-Item $stage -Recurse -Force }
}
