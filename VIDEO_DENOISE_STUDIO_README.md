# Video Denoise Studio 1.2.0

Video Denoise Studio is a separate Windows desktop application for temporal
video denoising. It reuses the proven media-probing, capability, preservation,
validation, cancellation, and recovery contracts from Deinterlace Studio
1.10.1, but it does **not** repair, deinterlace, resize, interpolate, or change
video cadence.

## What version 1.2.0 changes

Version 1.2.0 makes the validated six-denoiser catalog, parameter applicability,
CPU/GPU backend selection, FFmpeg/VapourSynth graph construction, media models,
probing, scheduling, output presets, validation, capability discovery, and
portable dependency support available through the neutral
`video_processing_core` package. Deinterlace Studio consumes the same shared
denoise/media implementation instead of maintaining a diverging copy.

Video Denoise Studio remains standalone and still does not repair or
deinterlace. Its portable build explicitly excludes repair modules and both
sibling GUI applications. Existing legacy app-local runtime folders are
recognized; new local installations use `Video Processing Runtime` beside the
executable. See `Shared Video Processing Core and Portability.md` in the
release folder.

The application has exactly two work areas:

1. **Single file + frame preview** — one source, a source-wide timeline,
   Topaz-style original/denoised frame comparison, and full-file processing.
2. **Batch processing** — an ordered queue of up to 99 sources, processed
   sequentially with synchronized denoise and output settings.

## Start and tool setup

Run `VideoDenoiseStudio.exe`. The portable executable is unsigned, so a
Windows reputation warning may appear on first launch.

The app invokes, but does not bundle:

- FFmpeg and FFprobe for probing, preview, encoding, and validation;
- VapourSynth and VSPipe for the four VapourSynth denoisers; and
- the corresponding capability-tested VapourSynth plugins.

On first launch, the app may read only the saved FFmpeg, FFprobe, and VSPipe
paths from Deinterlace Studio. It never changes Deinterlace Studio settings and
keeps its own preferences in:

`%APPDATA%\VideoDenoiseStudio\settings.json`

Use **Tools…** to inspect the resolved tools, versions, all six denoiser
capabilities, and selected CPU/GPU backends.

## Single-file frame-preview workflow

1. Choose or drop one source video.
2. Click or drag anywhere on the full video timeline to seek proportionally to
   that exact location, or use first, previous, next, last, Left Arrow, and
   Right Arrow navigation.
3. Leave **Frame preview** checked to render the selected target with the
   production denoiser. Uncheck it to render only the unprocessed target.
4. Compare the aligned images directly in the viewer:
   - press and hold the left mouse button for **Original**;
   - release it for **Denoised**;
   - drag while holding left to pan;
   - rotate the mouse wheel to zoom around the pointer; and
   - select **Fit** to reset zoom and pan.
5. Choose the codec, container, encoder-specific quality, and tracks.
6. Select **Process file**. Complete preflight runs automatically before any
   long encode starts; there is no separate **Check plan** button.

All six preserved-track controls share one compact row so the expanding
**Automatic preflight / run log** has more visible space.

Source-only rendering is asynchronous with no intentional debounce when
**Frame preview** is off. Denoised rendering remains briefly debounced so rapid
scrubbing does not launch a temporal graph for every pointer position. A newer
timeline or setting request supersedes older work, so a stale result cannot
replace the current target. Temporary preview files are removed when replaced
or when the app closes.

## Exact temporal preview

Only one target frame is displayed. The app applies the currently visible
denoiser and Strength, derives the hidden context from its normalized Temporal
radius, decodes those real neighboring frames, denoises the complete window,
and trims the result back to the target. There is no manual “48 frames” or
other preview-length guess. The completion status states the applied Strength,
radius, nominal window, real before/after context, and effective backend.

For example, radius 4 uses four real frames before plus the target plus four
after (9 total); radius 6 uses 13 total. At the source boundaries, the renderer
uses every neighbor that actually exists and reports the reduced before/after
counts. It never creates nonexistent context.

Changing Strength, radius, denoiser, Frame preview state, or refreshing the
same target preserves the exact current zoom and pan—even if the viewport was
changed while the replacement render was running. Selecting a different frame
or source resets the viewer to **Fit**, providing a predictable new-frame view.

Preview PNGs retain the source raster; Pillow performs high-quality display
scaling so zoom does not enlarge a pre-scaled proxy.

When available, exact source frame count is taken from stream metadata.
Matroska per-stream `DURATION` is preferred over a possibly longer audio or
container duration. If a source supplies neither frame count nor reliable
video duration, the last-frame address is a nominal average-frame-rate
estimate; ordinary CFR sources have exact frame addressing.

## Denoiser controls and applicability

**Strength 1–10 applies to all six denoisers**, but it is an app-normalized
artistic scale. Each number maps to a different native parameter:

| Denoiser | Native Strength mapping | Temporal-radius behavior |
|---|---|---|
| FFmpeg FFTDNOIZ | `sigma = Strength × 0.5` | Fixed radius 1; 3-frame window; control disabled |
| FFmpeg ATADENOISE | `A = Strength × 0.005`, `B = Strength × 0.01` | Radius 2–6; 5–13-frame odd window |
| VapourSynth V-BM3D | `sigma = 0.25 + Strength × 0.25` | Radius 1–6 |
| VapourSynth DFTTest2 | `sigma = Strength × 2.0` | Radius 1–3; installed CPU/NVRTC implementations allow up to 7 frames |
| VapourSynth MVTools | `thsad = 100 + Strength × 75` | Radius 1–6 |
| VapourSynth temporal NLMeans | `h = Strength × 0.3` | Radius 1–6 |

For each adjustable filter, radius `R` requests `R` real frames before and `R`
after the target: `2R + 1` total. ATADENOISE starts at radius 2 because FFmpeg
requires an odd window of at least five frames. The current optimized DFTTest2
CPU and NVRTC implementations accept radius 1–3, so the app prevents the
previously invalid 4–6 choices. FFTDNOIZ exposes only one previous and one next
frame in this route, so its radius is fixed at 1.

### Denoiser acceleration and selection ranks

Filter acceleration is automatic and capability-tested with a real bounded
graph. The current denoise panel reports the effective result as NVIDIA GPU
active, optimized CPU, CPU only, unavailable, or pending. V-BM3D and temporal
NLMeans prefer a verified CUDA route; DFTTest2 prefers an available verified
implementation; MVTools, FFTDNOIZ, and ATADENOISE are CPU filters in this app.
This is separate from NVENC output encoding and does not claim GPU video decode.

There is no acceleration on/off switch. A cosmetic switch could select
a route that capability discovery did not verify. Instead, the app chooses the
highest-priority route that successfully emitted a real test graph and tells
the user exactly which route is active.

The UI uses the following 1–6 selection guide, where 6 is highest/best:

| Denoiser | Quality | Speed | Practical character |
|---|---:|---:|---|
| V-BM3D | 6 | 1 | Quality-first reconstruction for difficult noise |
| DFTTest2 | 5 | 2 | Strong frequency-domain cleanup; NVRTC/JIT adds fresh-preview startup |
| MVTools | 4 | 3 | Motion-compensated temporal averaging |
| temporal NLMeans | 3 | 4 | Patch-similarity denoising; CUDA preferred here |
| ATADENOISE | 2 | 6 | Very fast adaptive temporal averaging |
| FFTDNOIZ | 1 | 5 | Fast fixed three-frame frequency-domain filtering |

The speed order was reproduced in two five-sample 960×540 preview-latency runs
on the validated local runtime: ATADENOISE 0.20 s, FFTDNOIZ 0.51 s,
NLMeans CUDA 0.84–0.86 s, MVTools 0.90–0.91 s, DFTTest2 NVRTC 1.66–1.68 s,
and V-BM3D CUDA 3.46–3.64 s. DFTTest2's NVRTC filter has much higher sustained
throughput than its CPU route, but a fresh one-frame preview pays process and
NVRTC/JIT startup. The displayed Speed score therefore describes interactive
one-frame preview responsiveness, not universal full-file throughput. Quality
ranks are engineering guidance, not an objective guarantee: content, motion,
noise, resolution, backend, Strength, and radius can change the useful order.
Compare several representative frames before a long run.

The **Denoiser guide…**, **Acceleration ?**, and adjacent **?** buttons explain
each filter, Strength mapping, radius behavior, selection ranks, backend,
preview context, and viewer gestures inside the app.

## Codec, quality, and container matrix

The quality control changes its name and valid range for the active encoder.
Controls that do not apply are disabled.

| Codec family | Effective encoder / quality | Valid containers | Recommended use |
|---|---|---|---|
| FFV1 16-bit | FFV1 v3 lossless; no CQ/CRF | MKV | Preservation or mastering; very large files |
| HEVC software | x265 placebo, CRF 0–51; optional x265 grain tune | MKV, MP4, MOV | Efficient delivery when NVENC is unavailable |
| HEVC NVIDIA | NVENC CQ 0–51, lower is better | MKV, MP4, MOV | Fast high-quality hardware delivery |
| AV1 software | libaom CRF 0–63 at `cpu-used 0`, or SVT-AV1 CRF at preset 0 | MKV, MP4 | Very efficient but potentially extremely slow |
| AV1 NVIDIA | NVENC CQ 0–63, lower is better | MKV, MP4 | Fast high-quality AV1 where playback supports it |
| ProRes 4444 XQ | Fixed `prores_ks` XQ profile; no CQ/CRF | MOV | High-bitrate editing intermediate |
| DNxHR 444 | Fixed `dnxhd` DNxHR 444 profile; no CQ/CRF | MOV | High-bitrate editing intermediate |

“Quality 14” never applies to FFV1, ProRes, or DNxHR. It is hidden behind a
disabled, accurately named control for those profiles. The x265 grain switch
applies only to software HEVC.

### NVIDIA HEVC and AV1

Hardware use is capability-gated by a real encode at the selected 10- or
12-bit route. The v1.1 NVENC contract is:

- P7 (slowest/best-quality preset);
- UHQ tuning;
- VBR constant quality with the selected CQ and target bitrate zero;
- full-resolution two-pass multipass;
- temporal AQ;
- middle B-frame reference mode; and
- no explicit lookahead count and no explicit temporal-filter depth.

UHQ itself enables lookahead and temporal filtering. The app therefore leaves
their depth to NVIDIA instead of duplicating or overriding UHQ. It also avoids
enabling spatial and temporal AQ together. The effective P7/UHQ route and
UHQ-managed lookahead are shown in the UI and retained run log. On the
validated local FFmpeg 9 / RTX PRO 6000 Blackwell route, both HEVC and AV1
reported automatic lookahead depth 25.

See NVIDIA's current [Video Codec SDK FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)
and [NVENC programming guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html).

## MKV, MP4, or MOV

The **Container** selector offers only combinations valid for the codec.
**Automatic** is source-stream aware:

- **MKV** is preservation-first, required for FFV1, the default for AV1, and
  recommended when retaining subtitles, attachments, data, or unusual audio.
- **MP4** is the broad-playback delivery choice for HEVC or AV1 when selected
  streams are compatible. HEVC is tagged `hvc1`, AV1 is tagged `av01`, and
  MP4 is written with fast-start metadata.
- **MOV** is required/recommended for ProRes and DNxHR editing; HEVC MOV is
  also available.

The source filename extension does not decide the output. Codec and selected
source streams do. Explicit MP4/MOV preflight refuses any selected audio,
subtitle, attachment, or data stream that cannot be preserved or safely
converted, and recommends MKV or deselecting that track type. Nothing is
silently dropped merely to make a file succeed.

Use **Codec + container guide…** and the output **?** buttons for this guidance
inside the app.

## Batch processing

Batch accepts files, folders, and optional folder trees. It rejects duplicate,
missing, unsupported, and over-capacity entries. It then:

1. probes and preflights every row before long work starts;
2. reserves noncolliding output paths;
3. shows each effective backend, codec, container, destination, and fallback;
4. streams preflight, encoder, validation, artifact-path, and result details to
   the scrolling **Batch run log** in the lower-left settings panel;
5. processes rows sequentially to bound CPU, GPU, RAM, and VRAM pressure; and
6. continues or stops after a row error according to the selected option.

If hardware is unavailable, Batch can use the matching software profile. If a
requested delivery profile cannot safely preserve a row, it can fall back to
16-bit native-chroma FFV1/MKV and records that decision in the row.

The first exact validation problem appears in a failed queue row. The complete
diagnostic and paths to the retained `.Denoise.log`, `.Denoise.json`, and any
quarantined output remain visible in the Batch run log.

## Preservation and interlaced sources

The app preserves the stored raster, sample/display aspect ratio, cadence,
duration/frame contract, color metadata, field state, and selected compatible
audio, subtitle, attachment, data, chapter, and metadata streams. Audio,
subtitles, attachments, chapters, and metadata default on; data is opt-in.

Interlaced stored frames stay interlaced. The app never bobs, runs
QTGMC/BWDIF, or changes cadence. Most denoisers operate on stored frames.
DFTTest2 is the precise exception because its wrapper accepts progressive
nodes only: for conclusive TFF/BFF input, the app temporarily separates the two
field parities, denoises each parity independently across the requested stored-
frame radius, and reweaves the original TFF/BFF frames. Opposite parities never
share a DFT temporal window. A no-filter identity test proves the split/reweave
is pixel-exact, and finished TFF/BFF files are validated for raster, frame count,
rate, duration, interlaced state, field order, and tracks. If field order cannot
be established, preflight stops instead of guessing. AV1 and the current NVENC
HEVC route are rejected for interlaced sources because they do not have a
verified field-preservation contract here.

## Safe processing and evidence

The source can never be its own output. Existing outputs and sidecars are
never overwritten. Every run encodes a unique hidden partial, validates it,
atomically promotes it, reopens the exact final file, validates again, and
computes SHA-256.

MP4/MOV `major_brand`, `minor_version`, and `compatible_brands` are structural
muxer identity fields, not portable user metadata. The app clears stale source
values so FFmpeg can regenerate brands appropriate to the new codec and does
not reject an otherwise correct output merely because these fields changed.
Titles, comments, chapters, selected tracks, and other meaningful metadata
remain subject to the normal strict preservation checks.

For `name.ext`, retained evidence is:

- `name.ext.Denoise.log` — command, automatic preflight warnings, encoder
  diagnostics, progress, and run result;
- `name.ext.Denoise.json` — complete plan, validation, elapsed time, output
  hash, and any rejected-file location; and
- `name.ext.Denoise.vpy` — the exact VapourSynth graph when used.

Cancellation stops active child processes, removes an unaccepted partial, and
never promotes it. A failed created file is quarantined under a unique
`.rejected.<token>` name when possible.

## Practical limitations

- Frame preview uses the real production filter and context, not a shortcut;
  difficult VapourSynth frames can take time.
- The timeline is a frame-seeking preview, not continuous real-time denoised
  playback.
- Source-boundary targets necessarily have less temporal context.
- CUDA, NVENC, and plugin availability depends on the actual drivers and
  binaries; a UI label never bypasses capability testing.
- Higher storage bit depth or 4:4:4 cannot restore detail absent from the
  source. Denoising quality remains content-dependent, so inspect several
  representative scenes before a long batch.

## Source launch, self-test, and reproducible build

```powershell
& 'work\pyinstaller311-env\Scripts\python.exe' denoise_main.py
```

```powershell
& 'release-denoise-v1.2.0-final\VideoDenoiseStudio.exe' --self-test `
  --self-test-report 'qa\denoise\packaged-selftest-v1.2.0.json' `
  --ffmpeg 'C:\path\to\ffmpeg.exe' `
  --ffprobe 'C:\path\to\ffprobe.exe' `
  --vspipe 'C:\path\to\vspipe.exe'
```

```powershell
& '.\build_denoise_release.ps1'
```

The hash-pinned build includes Python 3.11, Tcl/Tk, PyInstaller,
TkinterDnD2, and Pillow. FFmpeg, FFprobe, VapourSynth, VSPipe, denoiser
plugins, and NVIDIA components remain external.

## Release identity

Video Denoise Studio 1.2.0 is a non-overwriting successor to the frozen 1.1.4,
1.1.3, 1.1.2, 1.1.1, 1.1.0, and 1.0.0 releases and a separate application from
Deinterlace Studio 1.10.1. Their executables, settings, release directories,
and protected source checkpoints remain independent. The versioned release
record identifies the exact source composition, tests, package boundaries,
and release hash.
