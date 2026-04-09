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
$exePath = Join-Path $buildDir "Release\bin\$appName.exe"
$packageName = "$appName-windows-x64"
$packageRoot = Join-Path $distDir $packageName
$zipPath = Join-Path $distDir "$packageName.zip"
$requiredRuntimeDlls = @(
    "onnxruntime.dll",
    "abseil_dll.dll",
    "re2.dll",
    "libprotobuf-lite.dll",
    "libprotobuf.dll"
)
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

$catDogModelDir = Join-Path $workspaceRoot "models\cat_vs_dog"
if (Test-Path -LiteralPath $catDogModelDir) {
    $targetDir = Join-Path $packageRoot "models\cat_vs_dog"
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    foreach ($fileName in @("best.onnx", "labels.txt", "dataset_summary.json")) {
        $sourcePath = Join-Path $catDogModelDir $fileName
        if (Test-Path -LiteralPath $sourcePath) {
            Copy-Item -LiteralPath $sourcePath -Destination $targetDir
        }
    }
}

$wellColumnModelDir = Join-Path $workspaceRoot "models\WellColumnClassification"
if (Test-Path -LiteralPath $wellColumnModelDir) {
    $targetDir = Join-Path $packageRoot "models\WellColumnClassification"
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    foreach ($fileName in @("best.onnx", "labels.txt", "dataset_summary.json")) {
        $sourcePath = Join-Path $wellColumnModelDir $fileName
        if (Test-Path -LiteralPath $sourcePath) {
            Copy-Item -LiteralPath $sourcePath -Destination $targetDir
        }
    }
}

$runtimeBinDir = Join-Path $vcpkgRoot "installed\$vcpkgTriplet\bin"
if (-not (Test-Path -LiteralPath $runtimeBinDir)) {
    throw "Runtime bin directory not found: $runtimeBinDir"
}

foreach ($dllName in $requiredRuntimeDlls) {
    $sourcePath = Join-Path $runtimeBinDir $dllName
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required runtime DLL not found: $sourcePath"
    }

    Copy-Item -LiteralPath $sourcePath -Destination $packageRoot
}

$opencvDlls = @(Get-ChildItem -LiteralPath $runtimeBinDir -Filter "opencv*.dll")
if ($opencvDlls.Count -eq 0) {
    throw "OpenCV runtime DLLs not found in: $runtimeBinDir"
}

foreach ($dll in $opencvDlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination $packageRoot
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
