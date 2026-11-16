$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv venv --clear --python 3.11.14 .venv }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv }
    else { throw 'Install uv or Python 3.11 (py launcher).' }
}
if (Get-Command uv -ErrorAction SilentlyContinue) { uv pip install --python '.venv/Scripts/python.exe' -r requirements.txt }
else { & '.venv/Scripts/python.exe' -m pip install -r requirements.txt }
Write-Host 'Ready: .\.venv\Scripts\python.exe .\run.py predict'
