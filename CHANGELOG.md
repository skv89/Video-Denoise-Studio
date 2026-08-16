# Changelog

All notable changes to Video Denoise Studio are documented here.

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

