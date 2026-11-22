"""Build the English article from its editable Markdown source."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def escape(text):
    tokens = []
    def protect(match):
        tokens.append(match.group())
        return f"ZZTOKEN{len(tokens)-1}ZZ"
    text = re.sub(r"\$[^$]+\$|https://\S+", protect, text)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}
    text = "".join(replacements.get(c, c) for c in text)
    text = re.sub(r"\[(\d)\]", r"\\cite{r\1}", text)
    for index, token in enumerate(tokens):
        if token.startswith("https://"):
            token = r"\url{" + token + "}"
        text = text.replace(f"ZZTOKEN{index}ZZ", token)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    source = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    title = source.splitlines()[0][2:]
    preamble = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx,amsmath,amssymb,graphicx,booktabs,tabularx,placeins}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks,breaklinks=true]{hyperref}
\hypersetup{pdftitle={TITLE},pdfauthor={GAO Jiashen}}
\setlength{\parindent}{1.15em}
\setlength{\parskip}{2pt}
\setlength{\textfloatsep}{12pt}
\setlength{\intextsep}{12pt}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.08}
\emergencystretch=2em
\title{TITLE}
\author{GAO Jiashen\\\small 5003 Stochastic Processes and Applications}
\date{}
\begin{document}
\maketitle
'''.replace("TITLE", escape(title))
    blocks = source[source.index("## Abstract"):].split("\n\n")
    out = [preamble]
    abstract = False
    references = False
    table_number = 0
    table_captions = [
        "Training-set composition from generation 2 onward.",
        "Generation-2,000 performance averaged over five seeds.",
        "Generation-2,000 precision and recall under sampling bias.",
        "Initial and tail recall under a common fresh-data schedule.",
        "Tail squared sliced Wasserstein distance at a fixed fresh budget of 200."]
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("## ") or block.startswith("### "):
            if abstract:
                out.append(r"\end{abstract}")
                abstract = False
            heading = re.sub(r"^#+\s+(?:\d+(?:\.\d+)*\s+)?", "", block)
            if heading == "Abstract":
                out.append(r"\begin{abstract}\noindent")
                abstract = True
            elif heading == "References":
                out.append(r"\FloatBarrier\begin{thebibliography}{9}")
                references = True
            else:
                command = "subsection" if block.startswith("###") else "section"
                if heading == "Conclusion":
                    out.append(r"\clearpage")
                out.append("\\FloatBarrier\n\\" + command + "{" + escape(heading) + "}")
        elif references:
            match = re.match(r"\[(\d)\]\s+(.*)", block, flags=re.S)
            out.append(r"\bibitem{r" + match[1] + "}" + escape(match[2]))
        elif block.startswith("$$"):
            out.append("\\begin{equation}\n" + block[2:-2] + "\n\\end{equation}")
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[(.*)\]\((.*)\)", block, flags=re.S)
            filename = match[2].replace(".png", ".pdf")
            out.append(r"\begin{figure}[!htbp]\centering" + "\n"
                       + r"\includegraphics[width=\linewidth]{" + filename + "}\n"
                       + r"\caption{" + escape(match[1]) + "}\n" + r"\end{figure}")
        elif block.startswith("|"):
            rows = [line.strip().strip("|").split("|") for line in block.splitlines()]
            rows = [[cell.strip() for cell in row] for row in rows]
            columns = len(rows[0])
            table_number += 1
            caption = table_captions[table_number-1]
            out += [r"\begin{table}[!htbp]\centering\small", r"\caption{" + caption + "}",
                    r"\begin{tabularx}{\linewidth}{@{}" + "Y" * columns + r"@{}}\toprule",
                    " & ".join(r"\textbf{" + escape(c) + "}" for c in rows[0]) + r"\\\midrule"]
            out += [" & ".join(escape(c) for c in row) + r"\\" for row in rows[2:]]
            out += [r"\bottomrule\end{tabularx}\end{table}"]
        else:
            out.append(escape(block) + "\n")
    out += [r"\end{thebibliography}", r"\end{document}"]
    tex = "\n\n".join(out)
    (PAPER / "manuscript.tex").write_text(tex, encoding="utf-8")
    if args.compile:
        executable = shutil.which("pdflatex")
        if not executable:
            raise RuntimeError("pdflatex is required to compile manuscript.tex")
        (PAPER / "_build").mkdir(exist_ok=True)
        for _ in range(2):
            result = subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error",
                                     "-output-directory=_build", "manuscript.tex"],
                                    cwd=PAPER, capture_output=True, text=True, errors="replace")
            if result.returncode:
                raise RuntimeError(result.stdout[-6000:] + result.stderr[-2000:])
        shutil.copy2(PAPER / "_build/manuscript.pdf", PAPER / "manuscript.pdf")
        print("Compiled paper/manuscript.pdf")


if __name__ == "__main__":
    main()
