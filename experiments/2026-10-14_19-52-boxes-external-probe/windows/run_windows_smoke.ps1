param(
    [Parameter(Mandatory=$true)][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$ResultDirectory
)
$ErrorActionPreference = 'Stop'
$staging = Join-Path $env:TEMP ('autoboxes-engines-' + [Guid]::NewGuid().ToString('N'))
$diagnostics = Join-Path $staging 'diagnostics'
$child = $null
$code = 1
$statuses = @{}
New-Item -ItemType Directory -Path $diagnostics | Out-Null

function Get-VerifiedDownload([string]$Url, [string]$Destination, [string]$Sha256) {
    Write-Host ('Downloading pinned artifact: ' + [IO.Path]::GetFileName($Destination))
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination -TimeoutSec 60
    if ((Get-FileHash $Destination -Algorithm SHA256).Hash.ToLower() -ne $Sha256) {
        throw ('SHA256 mismatch: ' + $Destination)
    }
    Write-Host ('Verified SHA256: ' + [IO.Path]::GetFileName($Destination))
}

function Get-X86Runtimes([string[]]$Names) {
    $records = @()
    foreach ($name in $Names) {
        Write-Host ('Checking native x86 runtime: ' + $name)
        $runtime = Join-Path $env:SystemRoot ('SysWOW64\' + $name)
        if (-not (Test-Path -LiteralPath $runtime)) {
            throw ('Missing x86 runtime ' + $runtime + '; PRsBoxes needs MFC42, Dabble needs Microsoft Visual C++ 2013 x86. No engine moves attempted.')
        }
        $bytes = [IO.File]::ReadAllBytes($runtime)
        $pe = [BitConverter]::ToInt32($bytes, 0x3c)
        if ([BitConverter]::ToUInt16($bytes, $pe+4) -ne 0x14c) { throw ('Runtime must be x86: ' + $runtime) }
        $records += @{path=$runtime; sha256=(Get-FileHash $runtime -Algorithm SHA256).Hash.ToLower(); pe_machine='0x14c'}
    }
    return $records
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or -not [Environment]::Is64BitProcess) {
        throw 'Use 64-bit Windows PowerShell: System32\WindowsPowerShell\v1.0\powershell.exe'
    }
    foreach ($name in @('prsboxes_windows_smoke.py','dabble_windows_smoke.py','dabble_worker.py','positions.jsonl')) {
        Copy-Item (Join-Path $InputDirectory $name) $staging
    }
    Copy-Item (Join-Path $InputDirectory 'positions.jsonl') (Join-Path $diagnostics 'all-positions.jsonl')
    $allPositions = @(Get-Content (Join-Path $InputDirectory 'positions.jsonl') | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
    $selected = @()
    foreach ($positionId in @(1,3,6)) {
        $matches = @($allPositions | Where-Object { $_.position -eq $positionId })
        if ($matches.Count -ne 1) { throw ('Expected one frozen position with ID ' + $positionId) }
        $selected += ($matches[0] | ConvertTo-Json -Depth 10 -Compress)
    }
    [IO.File]::WriteAllLines((Join-Path $staging 'positions.jsonl'), [string[]]$selected, [Text.UTF8Encoding]::new($false))
    Copy-Item (Join-Path $staging 'positions.jsonl') $diagnostics
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $pythonUrl = 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip'
    $pythonSha = '8d3f33be9eb810f23c102f08475af2854e50484b8e4e06275e937be61ce3d2fb'
    $pythonZip = Join-Path $staging 'python.zip'
    Write-Host 'Preparing pinned native Windows Python; no Python installation required.'
    Get-VerifiedDownload $pythonUrl $pythonZip $pythonSha
    $pythonDirectory = Join-Path $staging 'python'
    Expand-Archive -LiteralPath $pythonZip -DestinationPath $pythonDirectory
    $python = Join-Path $pythonDirectory 'python.exe'
    $scriptHashes = @{}
    foreach ($name in @('prsboxes_windows_smoke.py','dabble_windows_smoke.py','dabble_worker.py','run_windows_smoke.ps1','run_windows_smoke.sh','positions.jsonl')) {
        $scriptHashes[$name] = (Get-FileHash (Join-Path $InputDirectory $name) -Algorithm SHA256).Hash.ToLower()
    }
    $provenance = [ordered]@{
        python_url=$pythonUrl; python_sha256=$pythonSha;
        python_hash_reference='https://github.com/UniClipboard/UniClipboard/blob/ab7204e85a331244c3f5032986a3c8fb38b7aa81/apps/gui-go/e2e/windows_native_panel_remote.py';
        python_cloud_verification='Cloud download policy denied python.org; pinned manifest digest is checked against bytes downloaded here.';
        input_sha256=$scriptHashes;
        selected_position_ids=@(1,3,6);
        selected_positions_sha256=(Get-FileHash (Join-Path $staging 'positions.jsonl') -Algorithm SHA256).Hash.ToLower();
        windows_version=[Environment]::OSVersion.VersionString;
        windows_64bit_process=[Environment]::Is64BitProcess;
        note='Three-root adapter smoke for each engine. Native Windows execution was not available in cloud Wine.'
    }
    $provenance | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $diagnostics 'runtime-provenance.json')
    $engines = @(
        @{name='prsboxes'; filename='PRsBoxes.exe'; worker='prsboxes_windows_smoke.py';
          url='https://raw.githubusercontent.com/yimmon/dots-and-boxes/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/other/PRsBoxes.exe';
          sha256='2f603782446e0df640b832966ac7b19e5ad9e8781aa6bc0c279fa415901b9d55';
          dlls=@('mfc42.dll'); extra=@('--limit','3','--watchdog','60')},
        @{name='dabble'; filename='dabble.exe'; worker='dabble_windows_smoke.py';
          url='https://raw.githubusercontent.com/ofenrohr/ksquares/3376c69a2b95cbc44241c65da224411967b32801/aux/dabble/dabble.exe';
          sha256='7e88069a551edae2fc8a404eca9fcb0366b3ba729738e0f4f5bea399a7704c32';
          dlls=@('mfc120.dll','MSVCR120.dll','MSVCP120.dll'); extra=@()}
    )
    foreach ($engine in $engines) {
        $name = $engine.name
        $child = $null
        $engineOutput = Join-Path $diagnostics $name
        New-Item -ItemType Directory -Path $engineOutput | Out-Null
        try {
            Write-Host ('Starting independent three-root smoke: ' + $name)
            $runtimeRecords = @(Get-X86Runtimes $engine.dlls)
            $exe = Join-Path $staging $engine.filename
            Get-VerifiedDownload $engine.url $exe $engine.sha256
            @{engine_url=$engine.url; engine_sha256=$engine.sha256; x86_runtimes=$runtimeRecords} |
                ConvertTo-Json -Depth 4 | Set-Content (Join-Path $engineOutput 'engine-provenance.json')
            $worker = Join-Path $staging $engine.worker
            $positions = Join-Path $staging 'positions.jsonl'
            $output = Join-Path $engineOutput 'roots'
            $arguments = @('-u', ('"' + $worker + '"'), '--exe', ('"' + $exe + '"'),
                           '--positions', ('"' + $positions + '"'), '--output', ('"' + $output + '"'),
                           '--seconds', '5') + $engine.extra
            Write-Host ('Running ' + $name + ' on frozen IDs 1, 3, 6; requested five seconds per decision.')
            $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $staging -PassThru `
                -RedirectStandardOutput (Join-Path $engineOutput 'worker.stdout.log') `
                -RedirectStandardError (Join-Path $engineOutput 'worker.stderr.log')
            # Retain the native handle while the process is alive. Windows
            # PowerShell's Start-Process object can otherwise lose ExitCode
            # after WaitForExit, turning a successful worker into null/"1".
            $null = $child.Handle
            # Independent per-engine watchdog covers hung window calls and descendants.
            if (-not $child.WaitForExit(300000)) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $child.Id /T /F | Out-Null
                throw 'Five-minute Windows worker watchdog expired; process tree stopped.'
            }
            $child.WaitForExit()
            $engineCode = $child.ExitCode
            if ($null -eq $engineCode) {
                throw 'Worker exited but Windows did not return its exit code; engine status is unknown. Inspect worker logs.'
            }
            $statuses[$name] = $engineCode
            Write-Host ($name + ' exit status: ' + $engineCode)
            Get-Content (Join-Path $engineOutput 'worker.stdout.log')
        } catch {
            ($_ | Out-String) | Set-Content (Join-Path $engineOutput 'launcher-error.txt')
            Write-Host ($name + ' stopped: ' + $_)
            $statuses[$name] = 1
        } finally {
            if ($null -ne $child -and -not $child.HasExited) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $child.Id /T /F | Out-Null
            }
            [string]$statuses[$name] | Set-Content (Join-Path $engineOutput 'exit-status.txt')
        }
    }
    $code = 0
    if (@($statuses.Values | Where-Object { $_ -ne 0 }).Count -gt 0) { $code = 1 }
} catch {
    ($_ | Out-String) | Set-Content (Join-Path $diagnostics 'bootstrap-error.txt')
    Write-Host ('Windows smoke bootstrap stopped: ' + $_)
    $code = 1
} finally {
    New-Item -ItemType Directory -Force -Path $ResultDirectory | Out-Null
    Copy-Item -Path (Join-Path $diagnostics '*') -Destination $ResultDirectory -Recurse -Force
    [string]$code | Set-Content (Join-Path $ResultDirectory 'exit-status.txt')
    $statuses | ConvertTo-Json | Set-Content (Join-Path $ResultDirectory 'engine-statuses.json')
    Write-Host ('Windows scratch directory: ' + $staging)
    Write-Host ('Collected diagnostics: ' + $ResultDirectory)
}
exit $code
