# Changelog

All notable changes to Video Denoise Studio are documented here.

## 1.1.4

- Fixed false MP4/MOV batch failures caused by treating muxer-generated
  `major_brand`, `minor_version`, and `compatible_brands` fields as portable
  user metadata. FFmpeg now regenerates those structural tags while meaningful
  source metadata remains strictly validated.
- Made retry naming reserve retained `.Denoise.log`, `.Denoise.json`, and
  `.Denoise.vpy` sidecars, preventing a previous failed run from forcing an
  unintended codec fallback.
- Added a persistent, scrolling Batch run log in the lower-left settings panel
  with live preflight, encoder, validation, failure, report, and rejected-file
  diagnostics for every row.
- Included the first exact validation error directly in failed queue rows while
  retaining complete details in the visible log and sidecars.
- Compacted the Batch denoiser panel by placing the quality/speed selection
  guide on the Strength/radius row beneath the denoiser selector.
- Standardized the packaged executable filename as `VideoDenoiseStudio.exe`.

## 1.1.3

- Added exact source-wide timeline seeking and automatic selected-frame
  preview rendering.
- Made preview context derive from the active denoiser's normalized temporal
  radius, including real source-boundary context reporting.
- Preserved viewer zoom and pan across Strength, radius, denoiser, refresh, and
  preview-state changes; selecting another frame or source resets to Fit.
- Added press-for-Original/release-for-Denoised comparison, panning, and
  pointer-centered wheel zoom.
- Added automatic capability-tested CPU, CUDA, and DFTTest2 NVRTC route
  selection with visible effective-backend status.
- Corrected DFTTest2 for progressive-only processing by splitting and reweaving
  conclusive interlaced field parities without cross-parity temporal mixing.
- Limited DFTTest2 radius to its supported 1–3 range; retained denoiser-specific
  radius constraints for the other filters.
- Added in-app denoiser descriptions, normalized Strength mappings,
  quality/speed selection guidance, and codec/container help.
- Added NVENC HEVC and AV1 P7/UHQ constant-quality profiles with
  capability-gated 10/12-bit routing and UHQ-managed lookahead.
- Made output quality labels and controls encoder-specific, disabling values
  that do not apply to FFV1, ProRes, or DNxHR.
- Added automatic preflight before processing and expanded the run-log area.
- Kept the application separate from Deinterlace Studio and protected the
  shared media/preservation core with its full regression suite.
