param([switch]$PlanOnly, [switch]$Resume)

$ErrorActionPreference = 'Stop'
$suiteRepo = Split-Path -Parent $PSScriptRoot
$suitePython = Join-Path $suiteRepo '.venv\Scripts\python.exe'
$suiteRoot = Join-Path $suiteRepo 'artifacts\casia-lmdb-full-loss-ph-seed7-20261019'
$suiteLossConfig = 'configs/experiment/loss_ablation_full_seed7.yaml'
$suitePhConfig = 'configs/experiment/stage2_ph_full_seed7.yaml'
$suiteLossSummary = Join-Path $suiteRepo 'experiments\casia-lmdb-full-loss-seed7-20261019\sweeps\casia-lmdb-full-loss-seed7-20261019\summary.json'
$suiteStatusPath = Join-Path $suiteRoot 'status.json'
Set-Location -LiteralPath $suiteRepo
New-Item -ItemType Directory -Path (Join-Path $suiteRoot 'logs') -Force | Out-Null

# PH first uses the bounded cache path. The independent loss study follows.
$suiteStages = @(
    @{ id = 'ph_projection'; requires = @(); arguments = @('-u', '-m', 'hccr.tda', 'project', '--config', $suitePhConfig) },
    @{ id = 'ph_extract'; requires = @('ph_projection'); arguments = @('-u', '-m', 'hccr.tda', 'extract', '--config', $suitePhConfig, '--splits', 'train', 'validation') },
    @{ id = 'ph_compare'; requires = @('ph_extract'); arguments = @('-u', '-m', 'hccr.tda', 'compare', '--config', $suitePhConfig) },
    @{ id = 'ph_test'; requires = @('ph_compare'); arguments = @('-u', '-m', 'hccr.tda', 'evaluate', '--config', $suitePhConfig, '--splits', 'test') },
    @{ id = 'loss_cross_entropy_arcface'; requires = @(); arguments = @('-u', 'scripts/run_experiments.py', '--config', $suiteLossConfig, '--resume', '--continue-on-error') }
)
$suiteSources = @(
    Get-ChildItem -LiteralPath (Join-Path $suiteRepo 'src\hccr') -Recurse -File -Filter '*.py'
    Get-Item -LiteralPath $suiteLossConfig, $suitePhConfig, $PSCommandPath
)
$suiteFingerprints = @($suiteSources | Sort-Object FullName | ForEach-Object {
    @{ path = $_.FullName; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
$suitePlanPath = Join-Path $suiteRoot 'queue_plan.json'
if (-not (Test-Path -LiteralPath $suitePlanPath) -or ($PlanOnly -and -not (Test-Path -LiteralPath $suiteStatusPath))) {
    [ordered]@{
        created_at = [DateTime]::UtcNow.ToString('o')
        repo = $suiteRepo; python = $suitePython
        classes = 7186; seed = 7; workers_per_image_loader = 4
        backend = 'existing_encoded_image_lmdb'; cuda_visible_devices = '0'
        ph_backbone = 'experiments/20260913T114748Z-d2e5e0a9'
        ph_controls = @('original', 'cnn', 'gap', 'flat', 'statistics', 'h0', 'h0_h1', 'mean_h0_h1')
        loss_variants = @('cross_entropy', 'arcface')
        holdout_policy = 'validation_selection_then_final_test_no_icdar_in_current_manifest'
        loss_free_commit_gate_gib = 24
        stages = $suiteStages; source_files = $suiteFingerprints
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $suitePlanPath -Encoding utf8
}
if ($PlanOnly) { Write-Output $suitePlanPath; exit 0 }

if (Test-Path -LiteralPath $suiteStatusPath) {
    if (-not $Resume) { throw 'Suite status exists. Inspect it before explicitly using -Resume.' }
    $suiteState = Get-Content -LiteralPath $suiteStatusPath -Raw | ConvertFrom-Json -AsHashtable
    if ($suiteState.status -eq 'running' -and (Get-Process -Id $suiteState.worker_pid -ErrorAction SilentlyContinue)) {
        throw 'The recorded suite worker is still running. Refusing a duplicate launch.'
    }
} else {
    $suiteState = [ordered]@{
        status = 'pending'; started_at = [DateTime]::UtcNow.ToString('o')
        worker_pid = $PID; active_stage = $null; child_pid = $null; stages = @()
    }
}
$suiteState.worker_pid = $PID
$suitePinnedPlan = Get-Content -LiteralPath $suitePlanPath -Raw | ConvertFrom-Json -AsHashtable
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'

function Save-SuiteState {
    $suiteState.updated_at = [DateTime]::UtcNow.ToString('o')
    $suiteTemporaryStatus = "$suiteStatusPath.tmp"
    $suiteState | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $suiteTemporaryStatus -Encoding utf8
    Move-Item -LiteralPath $suiteTemporaryStatus -Destination $suiteStatusPath -Force
}

try {
    foreach ($suiteStage in $suiteStages) {
        $suiteCompleted = @($suiteState.stages | Where-Object { $_.status -eq 'completed' } | ForEach-Object { $_.id })
        if ($suiteStage.id -in $suiteCompleted) { continue }
        $suiteRecord = [ordered]@{ id = $suiteStage.id; started_at = [DateTime]::UtcNow.ToString('o'); status = 'running' }
        $suiteState.stages += $suiteRecord
        if (@($suiteStage.requires | Where-Object { $_ -notin $suiteCompleted }).Count) {
            $suiteRecord.status = 'blocked_by_failed_dependency'
            Save-SuiteState
            continue
        }
        if ($suiteStage.id -eq 'loss_cross_entropy_arcface') {
            $suiteMemory = Get-CimInstance Win32_OperatingSystem
            $suiteRecord.free_commit_gib = $suiteMemory.FreeVirtualMemory / 1MB
            # Conservative launch policy, not a measured peak-memory requirement.
            $suiteRecord.required_free_commit_gib = $suitePinnedPlan.loss_free_commit_gate_gib
            if ($suiteRecord.free_commit_gib -lt $suiteRecord.required_free_commit_gib) {
                $suiteRecord.status = 'blocked_resources'
                $suiteRecord.reason = 'Full-manifest from-scratch training still duplicates Python row dictionaries across spawn workers. Free memory or optimize that path before resuming.'
                $suiteState.status = 'blocked_resources'
                Save-SuiteState
                break
            }
        }
        foreach ($suitePinnedFile in $suitePinnedPlan.source_files) {
            if ((Get-FileHash -LiteralPath $suitePinnedFile.path -Algorithm SHA256).Hash -ne $suitePinnedFile.sha256) {
                throw "Source/config changed since preflight: $($suitePinnedFile.path)"
            }
        }
        $suiteState.status = 'running'
        $suiteState.active_stage = $suiteStage.id
        $suiteRecord.stdout = Join-Path $suiteRoot "logs\$($suiteStage.id).stdout.log"
        $suiteRecord.stderr = Join-Path $suiteRoot "logs\$($suiteStage.id).stderr.log"
        $suiteChild = Start-Process -FilePath $suitePython -ArgumentList $suiteStage.arguments -WorkingDirectory $suiteRepo -WindowStyle Hidden -PassThru -RedirectStandardOutput $suiteRecord.stdout -RedirectStandardError $suiteRecord.stderr
        $suiteState.child_pid = $suiteChild.Id
        Save-SuiteState
        # Waiting happens only in this detached background worker, never in Codex.
        $suiteChild.WaitForExit()
        $suiteRecord.exit_code = $suiteChild.ExitCode
        $suiteRecord.status = if ($suiteChild.ExitCode -eq 0) { 'completed' } else { 'failed' }
        if ($suiteStage.id -eq 'loss_cross_entropy_arcface' -and $suiteRecord.status -eq 'completed') {
            $suiteLossResult = Get-Content -LiteralPath $suiteLossSummary -Raw | ConvertFrom-Json -AsHashtable
            $suiteRecord.loss_jobs = $suiteLossResult.jobs
            if ($suiteLossResult.status -ne 'completed') { $suiteRecord.status = 'failed' }
        }
        $suiteRecord.finished_at = [DateTime]::UtcNow.ToString('o')
        $suiteState.child_pid = $null
        Save-SuiteState
    }
    if ($suiteState.status -ne 'blocked_resources') {
        $suiteLatestStages = @($suiteState.stages | Group-Object { $_.id } | ForEach-Object { $_.Group[-1] })
        $suiteState.status = if (@($suiteLatestStages | Where-Object { $_.status -ne 'completed' }).Count) { 'completed_with_failures' } else { 'completed' }
    }
} catch {
    $suiteState.status = 'failed'
    $suiteState.error = $_.ToString()
} finally {
    $suiteState.active_stage = $null
    $suiteState.finished_at = [DateTime]::UtcNow.ToString('o')
    Save-SuiteState
}
if ($suiteState.status -ne 'completed') { exit 1 }
