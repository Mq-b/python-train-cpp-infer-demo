$ErrorActionPreference = "Stop"

$buildDir = $env:BUILD_DIR
$distDir = $env:DIST_DIR
$appName = $env:APP_NAME
$vcpkgRoot = $env:VCPKG_ROOT
$vcpkgTriplet = $env:VCPKG_TRIPLET

if ([string]::IsNullOrWhiteSpace($buildDir)) {
    throw "BUILD_DIR is not set."
}

if ([string]::IsNullOrWhiteSpace($distDir)) {
    throw "DIST_DIR is not set."
}

if ([string]::IsNullOrWhiteSpace($appName)) {
    throw "APP_NAME is not set."
}

if ([string]::IsNullOrWhiteSpace($vcpkgRoot)) {
    throw "VCPKG_ROOT is not set."
}

if ([string]::IsNullOrWhiteSpace($vcpkgTriplet)) {
    throw "VCPKG_TRIPLET is not set."
}

$workspaceRoot = Get-Location
$exePath = Join-Path $buildDir "bin\$appName.exe"
$packageName = "$appName-windows-x64"
$packageRoot = Join-Path $distDir $packageName
$zipPath = Join-Path $distDir "$packageName.zip"

if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Built executable not found: $exePath"
}

if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $packageRoot | Out-Null

Copy-Item -LiteralPath $exePath -Destination $packageRoot

$readmePath = Join-Path $workspaceRoot "README.md"
if (Test-Path -LiteralPath $readmePath) {
    Copy-Item -LiteralPath $readmePath -Destination $packageRoot
}

$modelPath = Join-Path $workspaceRoot "models\cat_vs_dog\best.onnx"
if (Test-Path -LiteralPath $modelPath) {
    $modelDir = Join-Path $packageRoot "models\cat_vs_dog"
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    Copy-Item -LiteralPath $modelPath -Destination $modelDir
}

$onnxBinDir = Join-Path $vcpkgRoot "installed\$vcpkgTriplet\bin"
if (Test-Path -LiteralPath $onnxBinDir) {
    Get-ChildItem -LiteralPath $onnxBinDir -Filter "onnxruntime*.dll" | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $packageRoot
    }
}

$windeployqt = Get-Command windeployqt.exe -ErrorAction Stop
& $windeployqt.Source `
    --release `
    --compiler-runtime `
    --no-translations `
    --no-opengl-sw `
    --dir $packageRoot `
    (Join-Path $packageRoot "$appName.exe")

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath
