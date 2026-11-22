# English manuscript

- `manuscript.pdf`: typeset article.
- `manuscript.md`: editable prose and tables.
- `manuscript.tex`: complete LaTeX source.
- `figures/`: eight numbered figures, in vector PDF and 300-DPI PNG.
- `data/results.json`: numerical summaries used by the article.

Compile the LaTeX source twice with `pdflatex manuscript.tex` from this folder.
In the full project, edit the Markdown source and use `code/reporting/build_manuscript.py --compile` to regenerate LaTeX and PDF.
