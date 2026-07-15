# Building the manuscript

`main.tex` is written for the freely available **IEEEtran** journal class, which
compiles anywhere TeX Live is installed:

```
make paper        # or, from paper/:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Converting to the official IEEE Access class

IEEE Access uses `ieeeaccess.cls`, which IEEE does not redistribute (download it
from the IEEE Author Center template package). To switch:

1. Drop `ieeeaccess.cls` and `ieeeaccess.bst` into this folder.
2. Replace the first line
   `\documentclass[journal]{IEEEtran}`
   with
   `\documentclass{ieeeaccess}`
   and add the Access front matter the template asks for (`\history{}`,
   `\doi{}`, `\corresp{}`, per-author `\IEEEmembership`).
3. Change `\bibliographystyle{IEEEtran}` to `\bibliographystyle{ieeeaccess}`.

The body, tables (`tables/*.tex`) and figures (`figures/*.png`) need no changes.

## Numbers are not hard-coded

Every headline number in the prose comes from `tables/macros.tex`, and every
table body from `tables/*.tex`. Both are produced by `scripts/make_tables.py`
from `results/*.json`. Re-run `make tables` after `make all` and the manuscript
updates itself. The shipped versions are synthetic smoke placeholders; see the
DRAFT note at the top of Section~V and remove it before submission.
