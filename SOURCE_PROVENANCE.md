# Source provenance

## Version 1.2.0

Video Denoise Studio 1.2.0 combines two independently preserved inputs:

1. the verified 2026-08-17 neutral-core implementation recorded by
   `SHARED_VIDEO_PROCESSING_CORE_RESUME_STATE.md` and
   `qa/shared-core-v1/final-release-manifest.json`; and
2. the complete v1.1.4 corrective changes preserved in local Git commit
   `e4ba1de9c9564d03fe562c98ae486ff98c082e9f`.

The selected denoise-only v1.2.0 source input contained 45 files and 504,284
bytes. The SHA-256 of its sorted `hash  relative/path` manifest was
`5AF0C4E80078D662EA23308C845CF152AB89CE2E9DB2374374D6FC5FD1A18F18`.
It consists of Video Denoise Studio, its tests and build inputs, plus exactly
15 `video_processing_core` denoise/media/runtime modules. Repair modules and
sibling GUI packages are excluded from both this public tree and the portable
build.

The original shared-core v1.2.0 artifact (SHA-256
`2AC502144B11AC9A83530AB48EFAC62988B3699D80707ED486BD56FF493A1278`)
proved the neutral-core architecture and package boundary, but it used the old
spaced filename and its source snapshot omitted the later v1.1.4 batch fixes.
It is therefore preserved as provenance evidence, not published as the final
GitHub v1.2.0 asset. The exact accepted public artifact hash is recorded in
`RELEASE_SHA256SUMS.txt`.

The carried-forward corrections prevent false MP4/MOV validation failures,
reserve retained diagnostic sidecars during retry naming, provide the visible
per-row Batch run log and exact failure detail, compact the Batch settings
layout, and retain the requested `VideoDenoiseStudio.exe` filename.

## Version 1.1.4

Video Denoise Studio 1.1.4 derives from the published 1.1.3 repository tree at
Git commit `6f8017a1bf9496b4d1cd9f87e94baba24ab4779e`. Its authorized change surface
is limited to Video Denoise Studio's MP4/MOV metadata planning, retry naming,
Batch diagnostics and layout, focused tests, versioning, documentation, and
release packaging. The protected Deinterlace Studio 1.10.1 shared core remains
unchanged.

The two retained 1.1.3 batch-failure reports demonstrated successful FFmpeg
encodes and exact frame counts followed only by false strict comparisons of
muxer-owned ISO-BMFF identity tags. Version 1.1.4 makes FFmpeg regenerate those
tags while continuing to validate meaningful metadata and the complete media
contract.

## Version 1.1.3 baseline

This repository tree reconstructs the source used by the verified Video
Denoise Studio 1.1.3 release from two immutable local checkpoints:

| Scope | Checkpoint | SHA-256 |
|---|---|---|
| Video Denoise Studio 1.1.3 app, tests, launcher, build file, and documentation | `source-v1.1.3.zip` | `30755613442276659679DA02B8B3268AF219E482421A38AB1AA8C3D2185F6CF1` |
| Protected Deinterlace Studio 1.10.1 shared core, entry point, and regression tests | `source-v1.1.2.zip` | `D5BA84A2B42897966ABAA949B5A8EE42A325A3FAD23DB227CDAFD2EFDD1AF346` |

The 1.1.3 release record identifies the frozen 1.1.2 source as its authoritative
baseline. Version 1.1.3 changed Video Denoise Studio layout, self-test guards,
and denoiser radius presentation; it did not authorize an upgrade of the
protected shared core.

The later-created 1.1.3 source ZIP cannot be used alone for a reproducible
public tree: its shared-core members report Deinterlace Studio 1.11.0 while its
included protected tests assert the 1.10.1 contracts. A clean run of that mixed
snapshot produced four shared-core failures. Restoring the explicitly named
1.1.2/Deinterlace 1.10.1 baseline yields the intended composition and passes:

- 51 Video Denoise Studio tests;
- 201 protected shared-core regression tests; and
- Python 3.11 compilation of both applications and both test suites.

Four occurrences of a developer-specific Windows username in a path-formatting
test fixture were changed to `ExampleUser`; this does not change application
code or test semantics.

The verified release executable distributed separately has SHA-256:

`F9B2D44B042762E59F382E8DAF41B006AB65257C57E01F9DAA47FF4907FD7C5E`
