$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$stateFile = Join-Path $workspace 'data/phone-demo.json'
if (-not (Test-Path -LiteralPath $stateFile)) { Write-Host 'No saved phone demo.'; exit }
$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
foreach ($demoId in @($state.dashboard_pid, $state.tunnel_pid)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$demoId" -ErrorAction SilentlyContinue
    $isDashboard = $process -and $process.Name -eq 'node.exe' -and $process.CommandLine -match "next.*start.*--port\s+$($state.port)(?:\s|$)"
    $isTunnel = $process -and $process.Name -eq 'cloudflared.exe' -and $process.CommandLine -match "--url\s+http://127\.0\.0\.1:$($state.port)(?:\s|$)"
    if ($isDashboard -or $isTunnel) { Stop-Process -Id $demoId }
}
Remove-Item -LiteralPath $stateFile
Write-Host 'Phone demo stopped. Desktop dashboard and FastAPI remain running.'
