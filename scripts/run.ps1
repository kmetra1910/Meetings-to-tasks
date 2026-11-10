<#
Run the pipeline in a container on a GPU host, local or remote through a Docker context.
Audio is converted to lossless 16 kHz mono FLAC locally, uploaded with a SHA256 check, and results are copied back
to results\<job>. The minutes come from -LlmUrl, or from a llama.cpp server started for this job when -LlmModel is set.
#>
param(
    [Parameter(Mandatory=$true)][string[]]$Files,
    # 0 estimates the number of voices; MaxSpeakers bounds the estimate.
    [int]$Speakers = 0,
    [int]$MaxSpeakers = 12,
    [string]$Title = '',
    # Empty detects the language from the first recording.
    [string]$Language = '',
    # Docker context of the GPU host; empty uses the current one.
    [string]$DockerContext = '',
    # GPU UUIDs this job may use; empty allows any GPU. Only an idle one is taken.
    [string[]]$AllowedGpus = @(),
    [string]$LlmUrl = 'http://localhost:8080/v1',
    # Path of a GGUF model on the Docker host; when set, a dedicated llama.cpp server runs on the same GPU for this job.
    [string]$LlmModel = '',
    [string]$LlmImage = 'ghcr.io/ggml-org/llama.cpp:server-cuda',
    [string[]]$LlmArgs = @('-ngl','99','-c','32768','-np','1','-fa','on','--jinja','--reasoning-budget','0')
)
$ErrorActionPreference = 'Stop'
$contextArgs = if ($DockerContext) { @('--context',$DockerContext) } else { @() }
# Windows PowerShell turns native stderr into errors; only exit codes mean failure here.
function Docker([string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    & docker.exe @contextArgs @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker failed: $($Arguments[0])" }
}
# The repository root: the image build context, and where data\, results\ and voices\ live.
$root = Split-Path $PSScriptRoot -Parent
$jobId = Get-Date -Format 'yyyyMMdd-HHmmss'
$container = "meetings-to-tasks-$jobId"
$inputDir = Join-Path $root "data\$jobId"
$outputDir = Join-Path $root "results\$jobId"
New-Item -ItemType Directory -Force -Path $inputDir,$outputDir | Out-Null
if ($DockerContext) {
    $context = (Docker @('context','inspect',$DockerContext) | ConvertFrom-Json)[0]
    # Recordings are private: refuse a remote daemon reached over TCP without TLS verification.
    $endpoint = $context.Endpoints.docker
    if ($endpoint.Host -like 'tcp://*' -and ($endpoint.SkipTLSVerify -or !$context.TLSMaterial.docker)) { throw "Docker context $DockerContext must use verified TLS" }
}
$gpuRows = Docker @('run','--rm','--gpus','all','nvidia/cuda:12.1.1-base-ubuntu22.04','nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits')
$processRows = Docker @('run','--rm','--gpus','all','nvidia/cuda:12.1.1-base-ubuntu22.04','nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader')
$busy = @($processRows | ForEach-Object { $_.Trim() })
$gpu = $gpuRows | ForEach-Object { $v=$_.Split(','); [pscustomobject]@{UUID=$v[0].Trim();Memory=[int]$v[1].Trim()} } |
    Where-Object { (!$AllowedGpus -or $_.UUID -in $AllowedGpus) -and $_.UUID -notin $busy } | Sort-Object Memory,UUID | Select-Object -First 1
if (!$gpu) { throw 'No permitted idle GPU. No work was launched.' }
$remoteFiles = @()
for ($i=0; $i -lt $Files.Count; $i++) {
    $source = (Resolve-Path -LiteralPath $Files[$i]).Path
    $target = Join-Path $inputDir "part$($i+1).flac"
    # Raw AAC recordings often end in a truncated frame; ffmpeg reports it but keeps the audio.
    $ErrorActionPreference = 'Continue'
    & ffmpeg -hide_banner -loglevel error -y -i $source -vn -ac 1 -ar 16000 -c:a flac $target
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg failed: $source" }
    $remoteFiles += "/work-data/part$($i+1).flac"
}
@{files=@($Files | ForEach-Object { Split-Path $_ -Leaf });gpu=$gpu.UUID;created=(Get-Date).ToString('o')} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $outputDir 'inputs.json')
# Build only this project's image; do not touch other workloads.
Docker @('build','-t','meetings-to-tasks:local',$root)
Docker @('run','-d','--name',$container,'--gpus',"device=$($gpu.UUID)",'--shm-size','2g','--entrypoint','sleep','meetings-to-tasks:local','infinity')
$llmContainer = $null
try {
    Docker @('cp',"$inputDir\.","${container}:/work-data")
    foreach ($remoteFile in $remoteFiles) {
        $localFile = Join-Path $inputDir ([System.IO.Path]::GetFileName($remoteFile))
        $expected = (Get-FileHash -LiteralPath $localFile -Algorithm SHA256).Hash.ToLowerInvariant()
        $actual = (Docker @('exec',$container,'sha256sum',$remoteFile)).Split(' ')[0]
        if ($actual -ne $expected) { throw "Incomplete upload: $remoteFile" }
    }
    $options = @('--out','/results','--speakers',"$Speakers",'--max-speakers',"$MaxSpeakers")
    if ($Title) { $options += @('--title',$Title) }
    if ($Language) { $options += @('--language',$Language) }
    if ($LlmModel) {
        # ASR first, so Whisper's GPU memory is free before the LLM loads on the same device.
        Docker (@('exec',$container,'m2t','run') + $remoteFiles + $options + @('--transcript-only'))
        $llmContainer = "$container-llm"
        $networkName = "$container-net"
        Docker @('network','create',$networkName)
        Docker @('network','connect',$networkName,$container)
        $modelDir = [System.IO.Path]::GetDirectoryName($LlmModel).Replace('\','/')
        $modelFile = [System.IO.Path]::GetFileName($LlmModel)
        # The model is mounted read-only and the server is reachable only inside the job network.
        Docker (@('run','-d','--name',$llmContainer,'--network',$networkName,'--gpus',"device=$($gpu.UUID)",'-e','CUDA_MODULE_LOADING=LAZY',
            '--mount',"type=bind,source=$modelDir,target=/models,readonly",$LlmImage,'-m',"/models/$modelFile",'--host','0.0.0.0','--port','8080') + $LlmArgs)
        Docker (@('exec',$container,'m2t','run','--summary-only','--llm-url',"http://${llmContainer}:8080/v1") + $options)
    } else {
        Docker (@('exec',$container,'m2t','run') + $remoteFiles + $options + @('--llm-url',$LlmUrl))
    }
    Docker @('cp',"${container}:/results/.",$outputDir)
    Write-Host "Ready: $outputDir\meeting.html"
} catch {
    # Preserve completed ASR checkpoints even if the minutes fail.
    & docker.exe @contextArgs cp "${container}:/results/." $outputDir
    throw
} finally {
    if ($llmContainer) {
        & docker.exe @contextArgs rm -f $llmContainer | Out-Null
        & docker.exe @contextArgs network rm "$container-net" | Out-Null
    }
    & docker.exe @contextArgs rm -f $container | Out-Null
}
if (Test-Path (Join-Path $root 'voices\library.json')) {
    # Suggest names from earlier confirmed meetings; a failure here must not lose the finished job.
    $ErrorActionPreference = 'Continue'
    Push-Location $root
    & python -m meetings_to_tasks voices match $outputDir
    Pop-Location
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Voice matching failed; name the speakers from speaker-samples as before' }
}
