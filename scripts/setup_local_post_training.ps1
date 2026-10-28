param(
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $taskRoot '.venv/Scripts/python.exe'
$receiptPath = Join-Path $OutputDirectory 'launcher.json'
$ownerPath = Join-Path $taskRoot 'artifacts/local-post-training-active.lock'
$ownerStream = [System.IO.File]::Open($ownerPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read)
$ownerBytes = [System.Text.Encoding]::UTF8.GetBytes("$PID")
$ownerStream.Write($ownerBytes, 0, $ownerBytes.Length)
$ownerStream.Flush()
Set-Location -LiteralPath $taskRoot
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$receipt = @{
    status = 'installing_dependencies'
    pid = $PID
    started_utc = [DateTime]::UtcNow.ToString('o')
    workspace = $taskRoot
    python = $pythonPath
    training = $false
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
try {
    & $pythonPath -m unittest tests.test_post_training_analysis tests.test_local_post_training tests.test_factorized_heads tests.test_classifier_quantization tests.test_gate_diagnostics -v *> (Join-Path $OutputDirectory 'tests.log')
    $receipt.test_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Local scientific checks failed; see tests.log. Analysis has not started.' }
    # Reuse the existing torch/CUDA installation; pip resolves only missing extras.
    & $pythonPath -m pip install 'timm>=1.0' 'onnx>=1.17' 'onnxruntime>=1.20' *> (Join-Path $OutputDirectory 'dependencies.log')
    $receipt.dependency_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) {
        # Native statistics, PyTorch inference and head-only PTQ remain useful.
        $receipt.dependency_warning = 'Some optional exports or modern baseline reconstruction may be blocked; see dependencies.log.'
    }
    $receipt.status = 'running_analysis'
    $receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
    $analysisOutput = Join-Path $OutputDirectory 'results'
    & $pythonPath -u scripts/run_local_post_training.py --output $analysisOutput *> (Join-Path $OutputDirectory 'run.log')
    $receipt.analysis_exit_code = $LASTEXITCODE
    $receipt.status = if ($LASTEXITCODE -eq 0) { 'completed' } else { 'completed_with_blockers' }
} catch {
    $receipt.status = 'failed'
    $receipt.error = $_.Exception.Message
} finally {
    $receipt.finished_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
    $ownerStream.Dispose()
    Remove-Item -LiteralPath $ownerPath
}
