param(
    [string]$ImageName = "phylo-local",
    [string]$ContainerName = "phylo-local-smoke",
    [int]$HostPort = 8000,
    [switch]$NoBuild,
    [switch]$KeepContainer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Cleanup-Container {
    param([string]$Name)
    try {
        $existing = docker ps -a --filter "name=^/$Name$" --format "{{.Names}}"
        if ($existing -eq $Name) {
            docker rm -f $Name | Out-Null
        }
    } catch {
        # Best effort cleanup.
    }
}

Write-Step "Checking Docker CLI"
docker --version | Out-Null

if (-not $NoBuild) {
    Write-Step "Building Docker image: $ImageName"
    docker build -t $ImageName .
} else {
    Write-Step "Skipping image build (--NoBuild)"
}

Write-Step "Ensuring no stale container named $ContainerName"
Cleanup-Container -Name $ContainerName

Write-Step "Starting container on localhost:$HostPort -> container:8000"
docker run -d --name $ContainerName -p "${HostPort}:8000" $ImageName | Out-Null

if (-not $KeepContainer) {
    Register-EngineEvent PowerShell.Exiting -Action {
        try { docker rm -f $using:ContainerName | Out-Null } catch {}
    } | Out-Null
}

Write-Step "Waiting for /health endpoint"
$healthUrl = "http://localhost:$HostPort/health"
$deadline = (Get-Date).AddMinutes(2)
$healthy = $false

while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 4
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
            $healthy = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $healthy) {
    Write-Host "`nContainer logs:" -ForegroundColor Yellow
    docker logs $ContainerName
    throw "Health check failed at $healthUrl"
}

Write-Host "Health check passed: $healthUrl" -ForegroundColor Green

Write-Step "Running in-container environment smoke test (PhyloEnvironment.reset)"
$pyCmd = @"
from server.phylo_environment import PhyloEnvironment
env = PhyloEnvironment(scene_type='BasicPulley', difficulty='EASY', seed=42, question_type='numeric')
obs = env.reset()
print('error=', obs.metadata.get('error', False))
print('problem_text_len=', len(obs.problem_text))
"@

docker exec $ContainerName python -c $pyCmd

Write-Host "`nSmoke test completed successfully." -ForegroundColor Green

if ($KeepContainer) {
    Write-Host "Container left running: $ContainerName" -ForegroundColor Yellow
    Write-Host "Stop it with: docker rm -f $ContainerName"
} else {
    Write-Step "Stopping and removing container"
    docker rm -f $ContainerName | Out-Null
    Write-Host "Container removed." -ForegroundColor Green
}
