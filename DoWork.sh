#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# DoWork.sh – venv → pip deps → Py scripts → TeX → PDF
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PAPERS_DIR="$ROOT_DIR/Papers"
LOGDIR="$ROOT_DIR/latex-logs"
MAIN_TEX="PBaikova-Paper"

DATA_CSV="$ROOT_DIR/Data/orlando_house_data.csv"
ROAD_GML="$ROOT_DIR/Data/orlando_road_network.graphml"

###############################################################################
# 1. Set up virtual environment with system Python
###############################################################################
PYTHON_EXEC=$(command -v python3)

echo "→ Using Python interpreter: $($PYTHON_EXEC --version)"

if [ ! -d "$VENV_DIR" ]; then
  echo "→ Creating virtual environment..."
  "$PYTHON_EXEC" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip

  # Get major.minor version (e.g., 3.9 or 3.11)
  PY_VER=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

  if [[ $(echo "$PY_VER < 3.10" | bc -l) -eq 1 ]]; then
    echo "→ Installing packages for Python < 3.10"
    "$VENV_DIR/bin/pip" install \
      numpy \
      scipy \
      pandas \
      matplotlib \
      shapely \
      geopandas \
      osmnx \
      scikit-learn \
      statsmodels \
      pygam
  else
    echo "→ Installing packages for Python ≥ 3.10"
    "$VENV_DIR/bin/pip" install \
      numpy \
      scipy \
      pandas \
      matplotlib \
      shapely \
      geopandas \
      osmnx \
      scikit-learn \
      statsmodels \
      pygam \
      jinja2
  fi
else
  echo "→ Virtual environment already exists; skipping creation."
fi


source "$VENV_DIR/bin/activate"

PYTHON="$(which python)"
echo "→ Activated virtual environment: $PYTHON"

###############################################################################
# 2. Input-file checks
###############################################################################
[[ -f "$DATA_CSV" ]] || { echo "ERROR: $DATA_CSV not found."; exit 1; }
[[ -f "$ROAD_GML" ]] || { echo "ERROR: $ROAD_GML not found."; exit 1; }

###############################################################################
# 3. Run analysis scripts
###############################################################################

PY_VER=$($PYTHON_EXEC -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "→ Detected Python version: $PY_VER"

if $PYTHON_EXEC -c "import sys; exit(not (sys.version_info >= (3, 10)))"; then
  echo "→ Running data-preparation-new.py (for Python ≥ 3.10)"
  "$VENV_DIR/bin/python" Code/data-preparation-new.py
else
  echo "→ Running data-preparation.py (for Python < 3.10)"
  "$VENV_DIR/bin/python" Code/data-preparation.py
fi
for script in database.py data-analysis.py; do
    echo "→ Running $script …"
    "$VENV_DIR/bin/python" -W ignore "$ROOT_DIR/Code/$script"
done
echo "✓ All analysis scripts finished."

###############################################################################
# 4. Assemble master TeX
###############################################################################
OUT_TEX="$ROOT_DIR/${MAIN_TEX}.tex"
echo "→ Building master TeX file..."
cat >"$OUT_TEX" <<'LATEX'
\documentclass[11pt]{article}
\usepackage{palatino}
\usepackage{setspace}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[C]{\thepage}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}\graphicspath{{Figures/}}
\usepackage{float}
\usepackage[authoryear]{natbib}
\usepackage{caption}
\usepackage{pdfpages}
\usepackage{booktabs}
\usepackage{url}
\IfFileExists{titlesec.sty}{\usepackage{titlesec}}{}
\IfFileExists{tocloft.sty}{\usepackage{tocloft}}{}
\floatstyle{ruled}
\doublespacing

\begin{document}
\begin{titlepage}
    \centering
    \vspace*{1in}
    {\Large University of Central Florida\par}
    \vspace{0.2in}
    {\large College of Business\par}
    {\large ECO6935 –25\par}
    {\large Capstone in Business Analytics\,I\par}
    \vspace{0.4in}
    {\large Polina\,Baikova\par}
    \vspace{1in}
    {\LARGE Understanding Housing Market Dynamics and Residential Property Valuation Patterns in Orlando, Florida Using Hedonic Price Modeling\par}
    \vfill
    {\large July\,2025\par}
\end{titlepage}
\setcounter{page}{1}
\newpage
\renewcommand{\thesection}{\arabic{section}}
LATEX

for TEX in \
  Introduction.tex Institutional-Details.tex Economic-Model.tex Statistical-Model.tex \
  Empirical-Specification.tex Empirical-Results.tex Recommendations.tex Limitations.tex \
  Conclusions.tex Acknowledgments.tex Appendix-A.tex Appendix-B.tex Appendix-C.tex Appendix-D.tex
do
  echo "\\input{Papers/$TEX}" >>"$OUT_TEX"
done

cat >>"$OUT_TEX" <<'LATEX'
\singlespacing
\nocite{*}
\bibliographystyle{chicago}
\bibliography{References/PBaikova-References}
\end{document}
LATEX
echo "✓ Master TeX created at $OUT_TEX"

###############################################################################
# 5. Compile PDF with LaTeX
###############################################################################
echo "==========================="
echo " Compiling PBaikova-Paper.tex..."
echo "==========================="

if [ -x /usr/bin/pdflatex ]; then PDFLATEX=/usr/bin/pdflatex
elif [ -x /usr/local/bin/pdflatex ]; then PDFLATEX=/usr/local/bin/pdflatex
else PDFLATEX=$(command -v pdflatex) || { echo "pdflatex not found – skipping"; exit 0; }
fi

if [ -x /usr/bin/bibtex ]; then BIBTEX=/usr/bin/bibtex
elif [ -x /usr/local/bin/bibtex ]; then BIBTEX=/usr/local/bin/bibtex
else BIBTEX=$(command -v bibtex) || { echo "bibtex not found – skipping"; exit 0; }
fi

echo "→ Using pdflatex: $PDFLATEX"
echo "→ Using bibtex:   $BIBTEX"

mkdir -p "$LOGDIR"
MAIN="PBaikova-Paper"
PASS_FAILED=0

run_latex() {
  local pass=$1
  if "$PDFLATEX" -interaction=nonstopmode -output-directory="$ROOT_DIR" "$MAIN.tex" > /dev/null 2>&1; then
    echo "✓ pdflatex pass $pass complete"
  else
    PASS_FAILED=1
  fi
}

run_latex 1

if [ -f "$MAIN.aux" ]; then
  AUX_PATH="$MAIN.aux"
else
  echo "⚠️  $MAIN.aux not found in current directory – searching recursively under $ROOT_DIR..."
  AUX_PATH=$(find "$ROOT_DIR" -type f -name "$MAIN.aux" | head -n 1)
fi

if [ -n "$AUX_PATH" ] && grep -q '\\citation' "$AUX_PATH"; then
  if "$BIBTEX" "$MAIN" > /dev/null 2>&1; then
    echo "✓ bibtex complete"
  else
    echo "✗ bibtex failed"
    PASS_FAILED=1
  fi
else
  echo "⚠️  No .aux file with citations found – skipping bibtex"
fi


run_latex 2
run_latex 3

if [ -f "$MAIN.pdf" ]; then
  echo "✅  PDF built → $MAIN.pdf"
  echo "   (logs in $LOGDIR)"
fi
