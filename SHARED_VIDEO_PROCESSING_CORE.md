# Shared Video Processing Core and Standalone Portability

The three desktop applications are developed from one neutral Python package,
`video_processing_core`. This is source-level reuse, not a requirement for a
separate installed service, DLL, application, or shared folder.

## Ownership

| Neutral module | Governing behavior | Used by |
|---|---|---|
| `media` | Stream models, probing, rational/aspect handling, schedules, codec presets, output validation, FFmpeg version parsing | All three apps as applicable |
| `repair` | Video Repair Tool issue taxonomy, packet/timestamp diagnosis, bounded decode samples, strict full-picture diagnosis, repair planning/profiles/execution, validation, rollback, hashes, atomic promotion | Video Repair Tool and Deinterlace Studio |
| `denoise` | The six validated Video Denoise Studio denoisers, parameter applicability, backend selection, FFmpeg/VapourSynth graph construction, and acceleration reporting | Video Denoise Studio and Deinterlace Studio |
| `runtime` | Safe tool discovery, capability probes, managed portable dependencies, and Windows file-drop support | All three apps as applicable |

Video Repair Tool remains authoritative for damage classification and repair.
Video Denoise Studio remains the reference workflow for shared denoiser and
general media-pipeline behavior. Application packages retain their own GUI,
settings, preview, queue, deinterlace policy, and workflow orchestration.

## Fast and strict diagnosis

Deinterlace Studio's normal analysis remains the fast path: a full compressed-
packet/timestamp pass plus bounded strict decode samples from the authoritative
repair scanner. It does not decode every picture. Before QTGMC processing or
when repair policy requires proof, both applications use the same strict
full-picture diagnosis entry point and the same stable issue identifiers.

## Portable builds do not depend on one another

Each PyInstaller executable embeds a private copy of only the neutral modules
it imports:

- Video Repair Tool includes shared `media`, `repair`, and repair-only runtime
  discovery. Its build explicitly excludes denoise and both sibling apps.
- Video Denoise Studio includes shared `media`, `denoise`, and runtime modules.
  Its build explicitly excludes repair and both sibling apps.
- Deinterlace Studio includes both shared repair and denoise functionality
  because its workflow can use both, but excludes both sibling GUI packages.

Consequently, users do not have to own all three tools, keep them together, or
install Python. Deleting or moving one executable does not disable another.
The code is common only in the development tree; the release executables are
self-contained.

New optional tool downloads are written beside the executable under
`Video Processing Runtime`. Read-only discovery also recognizes the former
`Deinterlace Studio Runtime`, `Video Repair Tool Runtime`, and
`Video Denoise Studio Runtime` folder names so a portable upgrade can reuse an
already validated local runtime. No system PATH or registry value is changed.

## Compatibility paths

Historical module paths remain as implementation-free aliases where needed for
settings, tests, or third-party scripts. They point to the neutral module
object; they are not second copies of the implementation.

## Release verification — 2026-08-20

The 2026-08-17 neutral-core integration baseline passed 216 Deinterlace Studio
tests, 61 Video Repair Tool tests, 54 protected repair tests, 51 Video Denoise
Studio tests, and 10 shared-core contract tests. The public Video Denoise
Studio v1.2.0 composition then carried forward four focused v1.1.4 regression
tests and passed all 55 application tests, compilation, recursive package
inspection, and its hidden packaged self-test.

Recursive PyInstaller archive inspection confirmed these boundaries:

- Video Repair Tool 1.8.0 contains 20 neutral modules, including the complete
  shared repair engine, and contains no denoise or sibling-application module.
- Deinterlace Studio 1.14.0 contains 23 neutral modules, including the shared
  repair and denoise feature families, and contains no sibling application.
- Video Denoise Studio 1.2.0 contains 15 neutral modules, including shared
  denoise, and contains no repair or sibling-application module.

Release executable SHA-256 values:

- Deinterlace Studio 1.14.0:
  `40162E740AB86C8004BCC8507657BA95A30A1CE0845BBA1FED17585E21A8E81A`
- Video Repair Tool 1.8.0:
  `23B13C07088DECB7E1278246E42B43553D20C6F6208FBA832C458BF6652F9FC9`
- Video Denoise Studio 1.2.0:
  `5500CDD36B35D99D197967C1B42A2AF4892922A1067B5AAEB36D04D01A7E377C`

## Residual boundaries

- The fast packet and sampled-decode scan is deliberately not represented as a
  full-picture guarantee. The shared strict decoded diagnosis remains the
  authority whenever final processing or repair policy requires proof.
- Optional GPU paths depend on the local driver and plugin runtime and retain
  verified CPU fallback. The current third-party Vulkan NNEDI3 graph still
  fails its safety probe on this system and therefore remains disabled.
- Each standalone executable embeds a private copy of the common modules it
  needs. This creates a small amount of packaging duplication, but never a
  sibling-tool requirement or an unused feature-family dependency.
- Historical compatibility aliases remain for import stability, but contain no
  implementation of their own.
