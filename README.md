# Video Denoise Studio

Video Denoise Studio 1.1.4 is a Windows desktop application dedicated to
temporal video denoising. It combines a source-wide timeline and a Topaz-style
original/denoised frame comparison with safe single-file and batch processing.

The app does not deinterlace, resize, interpolate, repair, or change cadence.
Its `deinterlace_studio` package is an internal shared media, capability,
preservation, validation, cancellation, and recovery core inherited from the
separately maintained Deinterlace Studio 1.10.1 reference base.

## Highlights

- Seek anywhere on the full video timeline.
- Automatically render the selected frame with the exact Strength and temporal
  radius chosen by the user.
- Derive the required real before/after context from the active denoiser—there
  is no manual preview-frame count.
- Hold the left mouse button for Original, release for Denoised, drag to pan,
  and use the mouse wheel to zoom.
- Preserve zoom and pan while comparing setting changes; reset to Fit only
  when selecting another frame or source.
- Choose among FFmpeg FFTDNOIZ, FFmpeg ATADENOISE, VapourSynth V-BM3D,
  DFTTest2, MVTools, and temporal NLMeans.
- Select the fastest capability-tested CPU or NVIDIA backend automatically and
  show the effective route in the UI.
- Process up to 99 queued sources sequentially with automatic preflight and a
  visible per-file Batch run log.
- Use preservation-oriented FFV1, delivery HEVC/AV1, or ProRes/DNxHR editing
  profiles with codec-aware container and quality controls.
- Validate a hidden partial before atomic promotion; never overwrite a source
  or existing output.

The complete operating guide, denoiser parameter mappings, quality/speed
selection guide, codec/container matrix, NVENC P7/UHQ contract, limitations,
and preservation behavior are in
[VIDEO_DENOISE_STUDIO_README.md](VIDEO_DENOISE_STUDIO_README.md).
The exact frozen-source composition is recorded in
[SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md).

## Download

The verified portable Windows executable is intended to be distributed as a
GitHub Release asset, not committed to the source tree. After publication,
download **VideoDenoiseStudio.exe** from this repository's Releases page and
verify it against [RELEASE_SHA256SUMS.txt](RELEASE_SHA256SUMS.txt).

The executable is currently unsigned, so Windows may show a reputation warning
on first launch.

## Runtime requirements

Video Denoise Studio invokes, but does not bundle:

- FFmpeg and FFprobe;
- VapourSynth and VSPipe for the four VapourSynth denoisers;
- the corresponding VapourSynth plugins; and
- optional NVIDIA drivers/components for capability-tested CUDA, NVRTC, and
  NVENC routes.

Open **Tools…** in the app to select and verify these external tools. Local app
preferences are stored in `%APPDATA%\VideoDenoiseStudio\settings.json`.

## Run from source

Use 64-bit Python 3.11 on Windows with a complete Tcl/Tk installation:

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes -r requirements-denoise-build.txt
& '.\.venv\Scripts\python.exe' denoise_main.py
```

FFmpeg/VapourSynth tool discovery remains external to the Python dependency
file. The drag-and-drop and Pillow dependencies are hash-pinned.

## Test

```powershell
& '.\.venv\Scripts\python.exe' -X dev -W error::ResourceWarning -m compileall -q denoise_main.py deinterlace_main.py video_denoise_studio video_denoise_tests deinterlace_studio deinterlace_tests
& '.\.venv\Scripts\python.exe' -X dev -W error::ResourceWarning -m unittest discover -s video_denoise_tests -v
& '.\.venv\Scripts\python.exe' -X dev -W error::ResourceWarning -m unittest discover -s deinterlace_tests -q
```

The protected shared-core regression suite is intentionally included because
the denoise application depends on those contracts.

## Build the Windows executable

```powershell
& '.\build_denoise_release.ps1'
```

The build script runs compilation and both test suites before packaging with
PyInstaller. It refuses to overwrite a nonempty versioned release directory.

## Privacy and data handling

Media processing is local. The app does not upload source videos. Outputs,
logs, JSON evidence, and VapourSynth graphs remain at user-selected local
paths.

## License status

No license has yet been selected for the original Video Denoise Studio source.
Until a project `LICENSE` file is added, copyright defaults apply and public
source availability alone does not grant reuse or redistribution rights.
Third-party components retain their own licenses; see
[VIDEO_DENOISE_STUDIO_THIRD_PARTY_NOTICES.md](VIDEO_DENOISE_STUDIO_THIRD_PARTY_NOTICES.md).
