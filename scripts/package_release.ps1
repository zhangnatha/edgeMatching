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

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'Missing required command: cmake' }
if ($OutputDir -eq [IO.Path]::GetPathRoot($OutputDir) -or $OutputDir -eq $repoDir -or
    ($OutputDir.StartsWith($repoDir + [IO.Path]::DirectorySeparatorChar) -and
     -not $OutputDir.StartsWith((Join-Path $repoDir 'dist') + [IO.Path]::DirectorySeparatorChar) -and
     $OutputDir -ne (Join-Path $repoDir 'dist'))) {
    throw "Refusing unsafe output directory: $OutputDir"
}

# 根据本地依赖判断是否构建 Qt 客户端。
$qtConfig = Join-Path $QtDir 'lib\cmake\Qt5\Qt5Config.cmake'
$qtEnabled = (-not $NoQt) -and (Test-Path $qtConfig)
$qtFlag = if ($qtEnabled) { 'ON' } else { 'OFF' }
$cmakeArgs = @('-S', $repoDir, '-B', $BuildDir, '-G', $Generator,
    '-A', 'x64', '-DCMAKE_BUILD_TYPE=Release',
    "-DBUILD_QT_CLIENT=$qtFlag")
if (Test-Path (Join-Path $OpenCVDir 'build\OpenCVConfig.cmake')) {
    $cmakeArgs += "-DOpenCV_DIR=$(Join-Path $OpenCVDir 'build')"
}
if ($qtEnabled) { $cmakeArgs += "-DCMAKE_PREFIX_PATH=$QtDir" }

Write-Host "Configuring release build (Qt client: $qtEnabled)..."
Invoke-Checked 'cmake' $cmakeArgs
Write-Host 'Building release binaries...'
Invoke-Checked 'cmake' @('--build', $BuildDir, '--config', 'Release', '--parallel')

$stage = Join-Path ([IO.Path]::GetTempPath()) ("edgeMatching-release-{0}" -f ([guid]::NewGuid()))
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
    $opencvBin = Join-Path $OpenCVDir 'bin'
    if (Test-Path $opencvBin) { Copy-Item (Join-Path $opencvBin '*.dll') $binDir -Force -ErrorAction SilentlyContinue }

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
