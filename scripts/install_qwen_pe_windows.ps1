param(
    [string]$ComfyUIRoot = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $ComfyUIRoot) {
    $customNodes = Split-Path -Parent $repoRoot
    $ComfyUIRoot = Split-Path -Parent $customNodes
}

$ComfyUIRoot = [IO.Path]::GetFullPath($ComfyUIRoot)
$modelDir = Join-Path $ComfyUIRoot "models\LLM\QwenImage2.1-PE"
$downloadDir = Join-Path $env:TEMP "xinbao-qwen-image-21-pe"
$runtimeDir = Join-Path $modelDir "runtime"
$sharedRuntimeDir = Join-Path $ComfyUIRoot "models\LLM\Bonsai2-27B\runtime"
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null

$artifacts = @(
    @{
        Name = "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
        Url = "https://hf-mirror.com/prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF/resolve/e18d4a3e0830ab157770738b16830e6fcf5f57d4/Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
        Size = 5629108864L
        Hash = "340FEB42C784E35B704A0F0D8D1C679A3FE60437BD9EA44A7A6FD2D730470506"
    },
    @{
        Name = "Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf"
        Url = "https://hf-mirror.com/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF/resolve/55b9c1a326599e142d59bcad8715d5601ccf8daa/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf"
        Size = 5629108864L
        Hash = "A5C6CB28CBAF838834D1335C8392619AF5809D9FE95D0EBD321E15751866DFE9"
    },
    @{
        Name = "Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
        Url = "https://hf-mirror.com/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF/resolve/55b9c1a326599e142d59bcad8715d5601ccf8daa/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
        Size = 921704608L
        Hash = "8DEDB71DBC3092DC47DE9108AD373D68A12854E2527D59BD9399601738F3BCE1"
    }
)

function Get-VerifiedArtifact([hashtable]$artifact) {
    $target = Join-Path $modelDir $artifact.Name
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target
        if ($item.Length -eq $artifact.Size -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $artifact.Hash) {
            Write-Host "Verified existing: $($artifact.Name)"
            return
        }
        throw "Existing model failed verification: $target"
    }

    $partial = Join-Path $downloadDir ($artifact.Name + ".part")
    Write-Host "Downloading or resuming $($artifact.Name)"
    & curl.exe --location --fail --retry 20 --retry-all-errors --retry-delay 5 `
        --continue-at - --output $partial $artifact.Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($artifact.Name)" }
    if ((Get-Item -LiteralPath $partial).Length -ne $artifact.Size) {
        throw "Size verification failed: $($artifact.Name)"
    }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $artifact.Hash) {
        throw "SHA256 verification failed: $($artifact.Name)"
    }
    Move-Item -LiteralPath $partial -Destination $target
}

foreach ($artifact in $artifacts) {
    Get-VerifiedArtifact $artifact
}

$promptCommit = "fb7ae1d1f9611cd91524d03c53c5246b36ac8577"
$prompts = @(
    @{
        Name = "system_prompt_t2i.txt"
        Url = "https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/$promptCommit/prompt_rewrite/prompts/system_prompt_t2i.txt"
        Hash = "A77C9A06C59B120741141D9514B95682BB8761D02BEC49CA61DEF7B2B3D9FB99"
    },
    @{
        Name = "system_prompt_edit.txt"
        Url = "https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/$promptCommit/prompt_rewrite/prompts/system_prompt_edit.txt"
        Hash = "E378FEA686A1431581BA4C654D332AE96ADAD633F144AE738EC8CE9C4FD66439"
    }
)

foreach ($prompt in $prompts) {
    $target = Join-Path $modelDir $prompt.Name
    if (-not (Test-Path -LiteralPath $target) -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $prompt.Hash) {
        & curl.exe --location --fail --retry 10 --output $target $prompt.Url
        if ($LASTEXITCODE -ne 0) { throw "Prompt download failed: $($prompt.Name)" }
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $prompt.Hash) {
        throw "Prompt SHA256 verification failed: $($prompt.Name)"
    }
}

if (Get-ChildItem -LiteralPath $sharedRuntimeDir -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue) {
    Write-Host "Reusing the existing Bonsai llama.cpp runtime: $sharedRuntimeDir"
} else {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $runtimeRelease = "https://github.com/ggml-org/llama.cpp/releases/download/b11068"
    $runtimeArchives = @(
        "llama-b11068-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip"
    )
    foreach ($archiveName in $runtimeArchives) {
        $archive = Join-Path $downloadDir $archiveName
        if (-not (Test-Path -LiteralPath $archive)) {
            & curl.exe --location --fail --retry 20 --retry-all-errors --retry-delay 5 `
                --output $archive "$runtimeRelease/$archiveName"
            if ($LASTEXITCODE -ne 0) { throw "Runtime download failed: $archiveName" }
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $runtimeDir -Force
    }

    if (-not (Get-ChildItem -LiteralPath $runtimeDir -Recurse -Filter llama-server.exe)) {
        throw "llama-server.exe was not found after runtime extraction."
    }
}

Write-Host "Qwen Image 2.1 PE installation complete: $modelDir" -ForegroundColor Green
Write-Host "Restart ComfyUI, then select a Qwen PE mode in 心宝❤推理（极速版）."
