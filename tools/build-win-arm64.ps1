[CmdletBinding()]
param(
    [string]$VenvDir = "",
    [string]$WheelDir = "",
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$VisualStudioPath = "",
    [string]$MsvcToolsetVersion = "14.51.36231",
    [string]$WindowsSdkVersion = "10.0.26100.0",
    [string]$CudaArchList = "12.0+PTX;10.3a",
    [string]$CMakeCudaArchitectures = "120f;103a",
    [ValidateRange(1, 256)]
    [int]$MaxJobs = 8,
    [string]$VersionOverride = "0.26.0+cu134",
    [switch]$SkipBuild
)

# This script expects VenvDir to be provisioned already. Install the local
# Windows ARM64 Torch wheel built for CUDA 13.4 separately; no compatible public
# PyPI Torch wheel is assumed. The remaining PyPI build packages are:
# cmake>=3.26.1, ninja, packaging>=24.2, setuptools>=77.0.3,<81.0.0,
# setuptools-scm>=8, setuptools-rust>=1.9.0, wheel, jinja2>=3.1.6, regex,
# build, and protobuf.
#
# Example:
# uv pip install --python "<venv>\Scripts\python.exe" `
#   "cmake>=3.26.1" ninja "packaging>=24.2" `
#   "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" `
#   "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6" regex build protobuf

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $VenvDir) {
    $VenvDir = Join-Path $RepoRoot ".venv"
}
if (-not $WheelDir) {
    $WheelDir = Join-Path $RepoRoot "dist\win-arm64-sm120-sm103-marlin-fa2"
}

$VenvDir = [IO.Path]::GetFullPath($VenvDir)
$WheelDir = [IO.Path]::GetFullPath($WheelDir)
$CudaOverlay = Join-Path $RepoRoot "build\cuda-include-arm64"
$ConfigureDir = Join-Path $RepoRoot "build\win-arm64-config"

function Find-VcVarsAll {
    if ($VisualStudioPath) {
        if (Test-Path -LiteralPath $VisualStudioPath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $VisualStudioPath).Path
        }
        $candidate = Join-Path $VisualStudioPath "VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        throw "VisualStudioPath does not contain vcvarsall.bat: $VisualStudioPath"
    }

    $roots = @(
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio"),
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    $candidate = foreach ($root in $roots) {
        Get-ChildItem (Join-Path $root "*\*\VC\Auxiliary\Build\vcvarsall.bat") `
            -File -ErrorAction SilentlyContinue |
            Where-Object {
                $installRoot = Split-Path -Parent (
                    Split-Path -Parent (
                        Split-Path -Parent (
                            Split-Path -Parent $_.FullName
                        )
                    )
                )
                Test-Path (
                    Join-Path $installRoot "VC\Tools\MSVC\$MsvcToolsetVersion"
                )
            }
    }

    $selected = $candidate |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if (-not $selected) {
        throw "Visual Studio ARM64 toolset $MsvcToolsetVersion was not found."
    }
    return $selected.FullName
}

function Import-Arm64MsvcEnvironment {
    $vcvars = Find-VcVarsAll
    $installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $command = (
        "set `"PATH=$installer;%PATH%`" && " +
        "call `"$vcvars`" arm64 -vcvars_ver=$MsvcToolsetVersion >nul && set"
    )
    $lines = & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "vcvarsall.bat failed with code $LASTEXITCODE"
    }

    foreach ($line in $lines) {
        $separator = $line.IndexOf("=")
        if ($separator -le 0) {
            continue
        }
        [Environment]::SetEnvironmentVariable(
            $line.Substring(0, $separator),
            $line.Substring($separator + 1),
            "Process"
        )
    }

    $cl = Get-Command cl.exe -ErrorAction Stop
    if ($cl.Source -notmatch "\\(HostARM64|Hostx64)\\arm64\\cl\.exe$") {
        throw "MSVC is not targeting ARM64: $($cl.Source)"
    }
    if ($cl.Source -notmatch "\\$([regex]::Escape($MsvcToolsetVersion))\\") {
        throw "MSVC selected the wrong toolset: $($cl.Source)"
    }
    return $cl.Source
}

function Initialize-CudaIncludeOverlay {
    $cudaHeader = Join-Path $CudaPath "include\cuda.h"
    $overlayHeader = Join-Path $CudaOverlay "cuda.h"
    $tensorMapStruct = "typedef struct CUtensorMap_st {"
    $alignmentDefinition = @"
#if defined(_MSC_VER)
#define TENSOR_MAP_ALIGN 64
#else
#define TENSOR_MAP_ALIGN 128
#endif

$tensorMapStruct
"@

    $content = [IO.File]::ReadAllText($cudaHeader)
    if ($content -notmatch "TENSOR_MAP_ALIGN") {
        if (
            -not $content.Contains($tensorMapStruct) -or
            -not $content.Contains("alignas(128)") -or
            -not $content.Contains("_Alignas(128)")
        ) {
            throw "CUDA tensor-map alignment declaration was not found."
        }
        $content = $content.Replace($tensorMapStruct, $alignmentDefinition)
        $content = $content.Replace(
            "alignas(128)",
            "alignas(TENSOR_MAP_ALIGN)"
        )
        $content = $content.Replace(
            "_Alignas(128)",
            "_Alignas(TENSOR_MAP_ALIGN)"
        )
    }

    New-Item -ItemType Directory -Force $CudaOverlay | Out-Null
    if (
        -not (Test-Path -LiteralPath $overlayHeader) -or
        [IO.File]::ReadAllText($overlayHeader) -cne $content
    ) {
        [IO.File]::WriteAllText(
            $overlayHeader,
            $content,
            [Text.UTF8Encoding]::new($false)
        )
    }
    $env:VLLM_CUDA_INCLUDE_OVERLAY = $CudaOverlay
}

function Set-BuildEnvironment {
    if (
        [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne
        [Runtime.InteropServices.Architecture]::Arm64
    ) {
        throw "This build requires native Windows ARM64."
    }
    if (-not $CudaPath) {
        throw "CUDA_PATH is not set. Pass -CudaPath explicitly."
    }

    $script:CudaPath = (Resolve-Path -LiteralPath $CudaPath).Path
    $python = Join-Path $VenvDir "Scripts\python.exe"
    $cmake = Join-Path $VenvDir "Scripts\cmake.exe"
    $ninja = Join-Path $VenvDir "Scripts\ninja.exe"
    foreach ($path in @($python, $cmake, $ninja)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required build tool is missing: $path"
        }
    }

    $cl = Import-Arm64MsvcEnvironment
    $vsToolsetRoot = Split-Path -Parent (
        Split-Path -Parent (
            Split-Path -Parent (
                Split-Path -Parent $cl
            )
        )
    )
    $sdkRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10"
    $sdkInclude = Join-Path $sdkRoot "Include\$WindowsSdkVersion"
    $sdkLib = Join-Path $sdkRoot "Lib\$WindowsSdkVersion"

    $requiredCudaFiles = @(
        "bin\nvcc.exe",
        "bin\ptxas.exe",
        "lib\arm64\cuda.lib",
        "lib\arm64\cudart.lib",
        "lib\arm64\cudart_static.lib",
        "lib\arm64\cublas.lib",
        "lib\arm64\cublasLt.lib",
        "lib\arm64\nvrtc.lib",
        "lib\arm64\cufftw.lib",
        "lib\arm64\nvml.lib",
        "lib\arm64\OpenCL.lib"
    )
    foreach ($relativePath in $requiredCudaFiles) {
        $path = Join-Path $CudaPath $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required ARM64 CUDA file is missing: $path"
        }
    }

    $env:PATH = (
        (Join-Path $VenvDir "Scripts"),
        (Join-Path $CudaPath "bin"),
        (Join-Path $sdkRoot "bin\$WindowsSdkVersion\arm64"),
        $env:PATH
    ) -join [IO.Path]::PathSeparator
    $env:INCLUDE = (
        (Join-Path $CudaPath "include"),
        (Join-Path $vsToolsetRoot "include"),
        (Join-Path $sdkInclude "ucrt"),
        (Join-Path $sdkInclude "shared"),
        (Join-Path $sdkInclude "um"),
        (Join-Path $sdkInclude "winrt"),
        (Join-Path $sdkInclude "cppwinrt")
    ) -join [IO.Path]::PathSeparator
    $env:LIB = (
        (Join-Path $CudaPath "lib\arm64"),
        (Join-Path $vsToolsetRoot "lib\arm64"),
        (Join-Path $sdkLib "ucrt\arm64"),
        (Join-Path $sdkLib "um\arm64")
    ) -join [IO.Path]::PathSeparator

    $env:CUDA_PATH = $CudaPath
    $env:CUDA_HOME = $CudaPath
    $env:CUDA_ROOT = $CudaPath
    $env:DISTUTILS_USE_SDK = "1"
    $env:VLLM_TARGET_DEVICE = "cuda"
    $env:TORCH_CUDA_ARCH_LIST = $CudaArchList
    $env:CMAKE_CUDA_ARCHITECTURES = $CMakeCudaArchitectures
    $env:MAX_JOBS = $MaxJobs.ToString()
    $env:NVCC_THREADS = "1"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $MaxJobs.ToString()
    $env:CMAKE_ARGS = (
        "-DCMAKE_SYSTEM_PROCESSOR=ARM64 " +
        "-DCMAKE_CUDA_ARCHITECTURES=$CMakeCudaArchitectures"
    )
    $env:VLLM_DISABLE_FA3_BUILD = "1"
    $env:VLLM_REQUIRE_RUST_FRONTEND = "0"
    $env:VLLM_DISABLE_SCCACHE = "1"
    $env:VLLM_VERSION_OVERRIDE = $VersionOverride

    Initialize-CudaIncludeOverlay

    $pythonInfo = & $python -c @"
import sysconfig
import torch

assert sysconfig.get_platform() == "win-arm64"
assert torch.version.cuda == "13.4"
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Python/Torch ARM64 validation failed."
    }
    $pythonInfo | Write-Host

    return @{
        Python = $python
        CMake = $cmake
        Ninja = $ninja
        Cl = $cl
    }
}

function Invoke-ConfigureOnly {
    param([hashtable]$Tools)

    $pythonPath = $Tools.Python.Replace("\", "/")
    $pythonSearchPath = (
        & $Tools.Python -c (
            "import sys; print(':'.join(" +
            "p.replace(chr(92), '/') for p in sys.path))"
        )
    ).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve the Python search path."
    }

    $cudaRoot = $CudaPath.Replace("\", "/")
    $cudaLib = "$cudaRoot/lib/arm64"
    $cl = $Tools.Cl.Replace("\", "/")
    $args = @(
        "-S", $RepoRoot,
        "-B", $ConfigureDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DVLLM_TARGET_DEVICE=cuda",
        "-DVLLM_PYTHON_EXECUTABLE=$pythonPath",
        "-DVLLM_PYTHON_PATH=$pythonSearchPath",
        "-DFETCHCONTENT_BASE_DIR:PATH=$($RepoRoot.Replace('\', '/'))/.deps",
        "-DNVCC_THREADS=1",
        "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
        "-DCMAKE_JOB_POOLS:STRING=compile=$MaxJobs",
        "-DCMAKE_C_COMPILER=$cl",
        "-DCMAKE_CXX_COMPILER=$cl",
        "-DCMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe",
        "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
        "-DCMAKE_CUDA_ARCHITECTURES=$CMakeCudaArchitectures",
        "-Dnvtx3_dir=$cudaRoot/include"
    )

    $cudaLibraries = @(
        @("CUDART_LIBRARY", "cudart.lib"),
        @("CUDA_CUDART", "cudart.lib"),
        @("CUDA_CUDART_LIBRARY", "cudart.lib"),
        @("CUDA_CUDA_LIBRARY", "cuda.lib"),
        @("CUDA_DRIVER_LIBRARY", "cuda.lib"),
        @("CUDA_NVRTC_LIB", "nvrtc.lib"),
        @("CUDA_OpenCL_LIBRARY", "OpenCL.lib"),
        @("CUDA_cublasLt_LIBRARY", "cublasLt.lib"),
        @("CUDA_cublas_LIBRARY", "cublas.lib"),
        @("CUDA_cuda_driver_LIBRARY", "cuda.lib"),
        @("CUDA_cudart_LIBRARY", "cudart.lib"),
        @("CUDA_cudart_static_LIBRARY", "cudart_static.lib"),
        @("CUDA_cufftw_LIBRARY", "cufftw.lib"),
        @("CUDA_nvml_LIBRARY", "nvml.lib"),
        @("CUDA_nvrtc_LIBRARY", "nvrtc.lib"),
        @("NVRTC_LIBRARY", "nvrtc.lib"),
        @("_CUBLASLT_LIBRARY", "cublasLt.lib"),
        @("_CUBLAS_LIBRARY", "cublas.lib")
    )
    foreach ($entry in $cudaLibraries) {
        $args += "-D$($entry[0]):FILEPATH=$cudaLib/$($entry[1])"
    }

    & $Tools.CMake @args
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed."
    }

    $markers = @{
        (Join-Path $RepoRoot ".deps\deepgemm-src\csrc\jit\compiler.hpp") =
            "cuobjdump.exe"
        (Join-Path $RepoRoot (
            ".deps\flashmla-src\csrc\kerutils\include\kerutils\device\common.h"
        )) = "using int128_t = int4;"
        (Join-Path $RepoRoot ".deps\vllm-flash-attn-src\CMakeLists.txt") =
            'list(APPEND FA2_ARCHS "12.0f")'
    }
    foreach ($entry in $markers.GetEnumerator()) {
        if (
            -not (Test-Path -LiteralPath $entry.Key) -or
            -not (Select-String -LiteralPath $entry.Key `
                -SimpleMatch $entry.Value -Quiet)
        ) {
            throw "Expected configure marker is missing: $($entry.Key)"
        }
    }

    Write-Host "Configuration completed successfully; build skipped."
}

$tools = Set-BuildEnvironment
git -C $RepoRoot diff --check
if ($LASTEXITCODE -ne 0) {
    throw "The source worktree has whitespace errors."
}

if ($SkipBuild) {
    Invoke-ConfigureOnly -Tools $tools
    exit 0
}

New-Item -ItemType Directory -Force $WheelDir | Out-Null
Push-Location $RepoRoot
try {
    & $tools.Python -m build --wheel --no-isolation --outdir $WheelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed."
    }
}
finally {
    Pop-Location
}

$wheel = Get-ChildItem $WheelDir -File -Filter "*.whl" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $wheel -or $wheel.Name -notmatch "win_arm64\.whl$") {
    throw "The build did not produce a Windows ARM64 wheel."
}

$hash = Get-FileHash $wheel.FullName -Algorithm SHA256
Write-Host "Built wheel: $($wheel.FullName)"
Write-Host "SHA256: $($hash.Hash)"
