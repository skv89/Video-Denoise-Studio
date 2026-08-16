# Source provenance for 1.1.3

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
