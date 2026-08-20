param(
    [string]$PythonExecutable = '',
    [string]$ReleaseDirectory = 'release-denoise-v1.2.0-final'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReleasePath = Join-Path $ProjectRoot $ReleaseDirectory
$WorkPath = Join-Path $ProjectRoot 'work\pyinstaller-denoise-v1.2.0'
$BuildEnvironment = Join-Path $ProjectRoot 'work\pyinstaller311-env'

if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $BuildEnvironment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonExecutable)) {
        $BasePython = 'C:\Program Files\Python311\python.exe'
        if (-not (Test-Path -LiteralPath $BasePython)) {
            throw 'Python 3.11 with a complete Tcl/Tk installation is required for the reproducible Windows build.'
        }
        & $BasePython -m venv $BuildEnvironment
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not create the workspace-local Python 3.11 build environment.'
        }
        & $PythonExecutable -m pip install 'PyInstaller==6.16.0'
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not install the pinned PyInstaller build dependency.'
        }
    }
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExecutable -c "import importlib.metadata; assert importlib.metadata.version('tkinterdnd2') == '0.6.2'; assert importlib.metadata.version('pillow') == '11.3.0'"
if ($LASTEXITCODE -ne 0) {
    & $PythonExecutable -m pip install --require-hashes -r (Join-Path $ProjectRoot 'requirements-denoise-build.txt')
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not install the hash-pinned TkinterDnD2 and Pillow build dependencies.'
    }
}
& $PythonExecutable -c "import tkinter, PyInstaller; r=tkinter.Tk(); r.withdraw(); print('Build runtime:', r.tk.call('info','patchlevel'), 'PyInstaller', PyInstaller.__version__); r.destroy()"
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python runtime cannot initialize Tk and PyInstaller: $PythonExecutable"
}
& $PythonExecutable -X dev -W error::ResourceWarning -m compileall -q denoise_main.py video_denoise_studio video_processing_core video_denoise_tests
if ($LASTEXITCODE -ne 0) {
    throw 'Compilation failed; packaging was stopped.'
}
& $PythonExecutable -X dev -W error::ResourceWarning -m unittest discover -s video_denoise_tests -v
if ($LASTEXITCODE -ne 0) {
    throw 'Video Denoise Studio tests failed; packaging was stopped.'
}
if ((Test-Path -LiteralPath $ReleasePath) -and (Get-ChildItem -LiteralPath $ReleasePath -Force | Select-Object -First 1)) {
    throw "The versioned release directory already contains files and will not be overwritten: $ReleasePath"
}
New-Item -ItemType Directory -Path $ReleasePath -Force | Out-Null
New-Item -ItemType Directory -Path $WorkPath -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $WorkPath 'userbase') -Force | Out-Null
$PreviousPythonUserBase = $env:PYTHONUSERBASE
try {
    $env:PYTHONUSERBASE = Join-Path $WorkPath 'userbase'
& $PythonExecutable -s -m PyInstaller --noconfirm --clean --onefile --windowed --hidden-import tkinter --hidden-import tkinterdnd2 --collect-all PIL --exclude-module video_processing_core.repair --exclude-module deinterlace_studio --exclude-module video_repair_tool --name 'VideoDenoiseStudio' --distpath $ReleasePath --workpath $WorkPath --specpath $WorkPath denoise_main.py
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller failed; no release was accepted.'
    }
}
finally {
    $env:PYTHONUSERBASE = $PreviousPythonUserBase
}
$ExecutablePath = Join-Path $ReleasePath 'VideoDenoiseStudio.exe'
$ArchiveListing = (& $PythonExecutable -m PyInstaller.utils.cliutils.archive_viewer -r -b $ExecutablePath | Out-String)
if ($ArchiveListing -notmatch 'video_processing_core\.denoise\.engine') {
    throw 'Package isolation check failed: the shared denoise engine is absent.'
}
if ($ArchiveListing -match 'video_processing_core\.repair|deinterlace_studio|video_repair_tool') {
    throw 'Package isolation check failed: Video Denoise Studio contains repair or sibling-application modules.'
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'VIDEO_DENOISE_STUDIO_README.md') -Destination (Join-Path $ReleasePath 'Video Denoise Studio README.md') -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'VIDEO_DENOISE_STUDIO_THIRD_PARTY_NOTICES.md') -Destination (Join-Path $ReleasePath 'Video Denoise Studio Third-Party Notices.md') -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'SHARED_VIDEO_PROCESSING_CORE.md') -Destination (Join-Path $ReleasePath 'Shared Video Processing Core and Portability.md') -Force
$PythonPrefix = (& $PythonExecutable -c "import sys; print(sys.prefix)").Trim()
$PillowLicense = Join-Path $PythonPrefix 'Lib\site-packages\pillow-11.3.0.dist-info\licenses\LICENSE'
if (-not (Test-Path -LiteralPath $PillowLicense)) {
    throw "The exact Pillow 11.3.0 combined license was not found: $PillowLicense"
}
Copy-Item -LiteralPath $PillowLicense -Destination (Join-Path $ReleasePath 'Pillow and bundled libraries license.txt') -Force
$ExecutableHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExecutablePath).Hash
$ChecksumPath = Join-Path $ReleasePath 'RELEASE_SHA256SUMS.txt'
[System.IO.File]::WriteAllText(
    $ChecksumPath,
    "$ExecutableHash *VideoDenoiseStudio.exe`n",
    [System.Text.UTF8Encoding]::new($false)
)
Get-FileHash -Algorithm SHA256 -LiteralPath (
    $ExecutablePath,
    $ChecksumPath,
    (Join-Path $ReleasePath 'Video Denoise Studio README.md'),
    (Join-Path $ReleasePath 'Video Denoise Studio Third-Party Notices.md'),
    (Join-Path $ReleasePath 'Shared Video Processing Core and Portability.md'),
    (Join-Path $ReleasePath 'Pillow and bundled libraries license.txt')
)
