# Video Denoise Studio 1.2.0 — Third-Party Notices

This notice distinguishes components packaged in `VideoDenoiseStudio.exe`
from external video-processing tools that the application invokes by path.

## Components packaged in the executable

### Python 3.11 and its standard library

Python is distributed under the Python Software Foundation License Version 2
and other compatible licenses covering included components. The authoritative
license history is available at:

https://docs.python.org/3/license.html

Copyright © 2001–2024 Python Software Foundation. All rights reserved.

### Tcl/Tk 8.6

The Python Windows distribution includes Tcl/Tk. Tcl/Tk is open-source
software distributed under BSD-style terms. Its license is available at:

https://www.tcl-lang.org/software/tcltk/license.html

### PyInstaller 6.16.0 bootloader and runtime hooks

Copyright © 2010–2023 PyInstaller Development Team.
Copyright © 2005–2009 Giovanni Bajo.
Based on previous work copyright © 2002 McMillan Enterprises, Inc.

PyInstaller is licensed under the GNU General Public License, version 2 or
later. Its Bootloader Exception grants unlimited permission to link or embed
the compiled bootloader and related files into combinations with other
programs and distribute those combinations without restriction arising from
those files. PyInstaller runtime hooks are licensed under Apache License 2.0.

https://pyinstaller.org/en/stable/license.html

### TkinterDnD2 0.6.2

MIT License

Copyright (c) 2020 Philippe Gagné

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/Eliav2/tkinterdnd2

### TkDND

This software is copyrighted by:

Georgios Petasis, Athens, Greece.

e-mail: petasisg@yahoo.gr, petasis@iit.demokritos.gr

Mac portions (c) 2009-2014 Kevin Walzer/WordTech Communications LLC,
kw@codebykevin.com

The following terms apply to all files associated with the software unless
explicitly disclaimed in individual files.

The authors hereby grant permission to use, copy, modify, distribute, and
license this software and its documentation for any purpose, provided that
existing copyright notices are retained in all copies and that this notice is
included verbatim in any distributions. No written agreement, license, or
royalty fee is required for any of the authorized uses.

Modifications to this software may be copyrighted by their authors and need
not follow the licensing terms described here, provided that the new terms are
clearly indicated on the first page of each file where they apply.

IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
OF THE USE OF THIS SOFTWARE, ITS DOCUMENTATION, OR ANY DERIVATIVES THEREOF,
EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE, AND NON-INFRINGEMENT. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

GOVERNMENT USE: If you are acquiring this software on behalf of the U.S.
government, the Government shall have only "Restricted Rights" in the software
and related documentation as defined in the Federal Acquisition Regulations
(FARs) in Clause 52.227.19 (c) (2). If you are acquiring the software on behalf
of the Department of Defense, the software shall be classified as "Commercial
Computer Software" and the Government shall have only "Restricted Rights" as
defined in Clause 252.227-7013 (c) (1) of DFARs. Notwithstanding the foregoing,
the authors grant the U.S. Government and others acting in its behalf
permission to use and distribute the software in accordance with the terms
specified in this license.

### Pillow 11.3.0

Pillow supplies source-raster image loading and high-quality viewer scaling,
zoom, and pan. Pillow is the friendly PIL fork and is licensed under the
open-source MIT-CMU License.

The Python Imaging Library (PIL) is copyright © 1997–2011 Secret Labs AB and
copyright © 1995–2011 Fredrik Lundh and contributors. Pillow is copyright
© 2010 Jeffrey A. Clark and contributors.

The release includes `Pillow and bundled libraries license.txt`, copied
verbatim from the exact Pillow 11.3.0 wheel. That file contains the complete
Pillow license and notices for bundled image libraries, including Brotli,
FreeType, libjpeg-turbo, libpng, libwebp, OpenJPEG, LibTIFF, XZ Utils, and
zlib-ng.

https://python-pillow.org/

## External processing tools and plugins (not packaged)

`VideoDenoiseStudio.exe` does not contain, install, download, modify, or
redistribute FFmpeg, FFprobe, VapourSynth, VSPipe, NVIDIA runtimes, or
VapourSynth denoiser plugins. It invokes binaries selected by the user or read
from an existing Deinterlace Studio configuration. Those installations retain
their own licenses and notices.

Depending on the selected denoiser and available backend, a compatible
external runtime may include:

- FFmpeg/FFprobe, including a GPL build such as the Gyan full build;
- VapourSynth and VSPipe;
- BestSource;
- VSJetpack Python packages including `vstools` and `vsdenoise`;
- V-BM3D CPU, CUDA, CUDA RTC, or `vszipcu` implementations;
- DFTTest2 CPU, NVIDIA NVRTC, or NVIDIA cuFFT implementations;
- MVTools and temporal NLMeans implementations; and
- NVIDIA CUDA/NVRTC/cuFFT and display-driver components where applicable.

Representative upstream license information inherited from the compatible
Deinterlace Studio runtime contract includes:

- Gyan FFmpeg full build: GPLv3; individual FFmpeg components retain their
  upstream licenses.
- VapourSynth portable release: LGPL-2.1-or-later.
- Python embedded distribution used by an external runtime: Python Software
  Foundation License.
- VSJetpack Python packages: MIT; individual native plugins retain their own
  upstream licenses.
- VapourSynth-BM3DCUDA: GPL-2.0-or-later.
- VapourSynth-DFTTest2 CPU and optional NVIDIA implementations: GPL-3.0.
- VapourSynth-vszipcu: MIT.
- NVIDIA CUDA runtime components: NVIDIA proprietary license.

Consult the notices included with the exact external runtime and the upstream
projects before redistribution. The external toolchain is not part of the
Video Denoise Studio executable or this release directory.

## No warranty from third-party authors

Third-party components are provided under their respective licenses and
without warranty. Nothing in this notice changes or supersedes those licenses.
