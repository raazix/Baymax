param([ValidateRange(1024,65535)][int]$Port = 3002, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dashboardDirectory = Join-Path $workspace 'apps/dashboard'
$stateFile = Join-Path $workspace 'data/phone-demo.json'
$tunnelCommand = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $tunnelCommand) { throw 'Install cloudflared first: winget install Cloudflare.cloudflared' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is in use. Stop the previous phone demo or choose -Port 3003." }
if (-not $SkipBuild) {
    Push-Location $dashboardDirectory
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Dashboard build failed.' } }
    finally { Pop-Location }
}
if (-not (Test-Path (Join-Path $dashboardDirectory '.next/BUILD_ID'))) { throw 'Build the dashboard before using -SkipBuild.' }
New-Item -ItemType Directory -Path (Join-Path $workspace 'data') -Force | Out-Null
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$pinBytes = New-Object byte[] 4; $secretBytes = New-Object byte[] 32
$rng.GetBytes($pinBytes); $rng.GetBytes($secretBytes); $rng.Dispose()
$pairingCode = (([BitConverter]::ToUInt32($pinBytes, 0) % 900000) + 100000).ToString()
$previousCode = $env:LINEGUARD_PHONE_ACCESS_CODE; $previousSecret = $env:LINEGUARD_PHONE_SESSION_SECRET
$dashboardProcess = $null; $tunnelProcess = $null
try {
    $env:LINEGUARD_PHONE_ACCESS_CODE = $pairingCode
    $env:LINEGUARD_PHONE_SESSION_SECRET = [Convert]::ToBase64String($secretBytes)
    $dashboardProcess = Start-Process -FilePath 'node.exe' -ArgumentList 'node_modules/next/dist/bin/next','start','--hostname','127.0.0.1','--port',$Port -WorkingDirectory $dashboardDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $workspace 'data/phone-next.out.log') -RedirectStandardError (Join-Path $workspace 'data/phone-next.err.log')
    $env:LINEGUARD_PHONE_ACCESS_CODE = $previousCode; $env:LINEGUARD_PHONE_SESSION_SECRET = $previousSecret
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if ($dashboardProcess.HasExited) { throw 'Phone dashboard failed to start; inspect data/phone-next.err.log.' }
        try { $response = Invoke-WebRequest "http://127.0.0.1:$Port/phone-access" -UseBasicParsing -TimeoutSec 3; if ($response.StatusCode -eq 200) { $ready = $true; break } } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'Phone dashboard did not become ready.' }
    $tunnelLog = Join-Path $workspace 'data/phone-tunnel.err.log'
    $tunnelProcess = Start-Process -FilePath $tunnelCommand.Source -ArgumentList '--no-autoupdate','tunnel','--url',"http://127.0.0.1:$Port",'--protocol','http2' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $workspace 'data/phone-tunnel.out.log') -RedirectStandardError $tunnelLog
    $phoneUrl = $null
    for ($attempt = 0; $attempt -lt 75; $attempt++) {
        if ($tunnelProcess.HasExited) { throw 'Phone tunnel failed; inspect data/phone-tunnel.err.log.' }
        if (Test-Path $tunnelLog) { $logText = Get-Content -LiteralPath $tunnelLog -Raw -ErrorAction SilentlyContinue; if ($logText) { $match = [regex]::Match($logText, 'https://[a-z0-9-]+\.trycloudflare\.com'); if ($match.Success) { $phoneUrl = $match.Value; break } } }
        Start-Sleep -Milliseconds 1000
    }
    if (-not $phoneUrl) { throw 'No HTTPS phone link was returned. Check the tunnel log.' }
    $state = @{ url=$phoneUrl; code=$pairingCode; expires_at=[DateTime]::UtcNow.AddHours(8).ToString('o'); port=$Port; dashboard_pid=$dashboardProcess.Id; tunnel_pid=$tunnelProcess.Id }
    [IO.File]::WriteAllText($stateFile, ($state | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Write-Host "Phone workspace: $phoneUrl"
    Write-Host "Pairing code: $pairingCode"
    Write-Host 'Open Connect phone on the laptop for the QR code. Keep FastAPI and the laptop running.'
    Write-Host 'Stop this demo with: powershell -File scripts/stop_phone_demo.ps1'
} catch {
    if ($tunnelProcess -and -not $tunnelProcess.HasExited) { Stop-Process -Id $tunnelProcess.Id }
    if ($dashboardProcess -and -not $dashboardProcess.HasExited) { Stop-Process -Id $dashboardProcess.Id }
    throw
} finally {
    $env:LINEGUARD_PHONE_ACCESS_CODE = $previousCode; $env:LINEGUARD_PHONE_SESSION_SECRET = $previousSecret
}
