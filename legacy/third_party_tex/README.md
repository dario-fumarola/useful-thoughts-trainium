# Historical LaTeX dependencies

These unchanged upstream files accompany the third-party styles already present
in `../ema-trainium-pre-thinking/overleaf/iclr2027/`. They are redistribution
sources and documentation, not active training code. No files in the original
research snapshot have been replaced.

| Material | Author and terms | Included files |
| --- | --- | --- |
| natbib 8.31, 2009-07-16 | Patrick W. Daly; LPPL version 1 or later | `natbib-8.31/`, including `natbib.dtx`, `natbib.ins`, the package manifest, documentation, bibliography styles, and bibentry sources |
| fancyhdr 3.2, 2005-03-22 | Piet van Oostrum; LPPL version 1 or later | `fancyhdr-3.2/`, including the original `.sty` sources, `README`, and documentation |
| License text | LaTeX3 Project, 1999 | [`LPPL-1.0.txt`](LPPL-1.0.txt), copied unchanged from the [LaTeX Project](https://www.latex-project.org/lppl/lppl-1-0.txt) |

[`PROVENANCE.json`](PROVENANCE.json) lists every copied package file, its SHA256,
and its original archive path. Together with the package manifests and retained
headers, it identifies the files covered by their respective license notices.

The sources come from the [official TeX Live 2009 release archive](https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2009/texlive-20091107-texmf.tar.xz),
hosted in the historic TeX archive. The complete archive was verified against its
[published SHA256](https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2009/texlive-20091107-texmf.tar.xz.sha256):
`63e38a218c04e28081e12b4f48969a7e25cf5f81a8c4807aae815a49b575ae5d`.
The preserved `natbib.sty` and `fancyhdr.sty` are each byte-identical to the
corresponding file in the research snapshot.

Natbib's generated-file notice requires its original `natbib.dtx` when
redistributing `natbib.sty`; that exact version is supplied here. The archived
`README.v831` lists the original package contents. Fancyhdr 3.2 is itself source;
its original README explains that this release predates a `.dtx` distribution.
Its companion sources and documentation are included together.

The research snapshot's `iclr2027_conference.bst` retains its LPPL notice and
attributes its adaptations to Hal Daumé III, J. Fürnkranz, and Patrick W. Daly.
The unmodified natbib `plainnat.bst` family is included here; the current
[natbib CTAN page](https://ctan.org/pkg/natbib) also provides upstream availability.
No additional source-distribution condition is stated in that adapted `.bst`.
The conference `.sty` retains its Hugo Larochelle/NIPS/CVPR attribution; this
repository assigns it no additional license.

These package licenses do not license the surrounding research code, papers,
or project history. See the repository [`NOTICE.md`](../../NOTICE.md).
