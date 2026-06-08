#!/usr/bin/env bash
# Compila la memoria del TFM de forma reproducible.
# biblatex (backend=bibtex) + hyperref pueden necesitar varias invocaciones en un
# árbol limpio para que las referencias cruzadas y el outline PDF converjan; por
# eso se repite latexmk hasta que el log queda limpio (máx. 4). Uso: ./build.sh
set -u
cd "$(dirname "$0")"

clean_log() {
  local undef fatal
  undef=$(grep -iE 'undefined (reference|citation)' main.log 2>/dev/null | wc -l | tr -d ' ')
  fatal=$(grep -E 'BKM@entry|@@BOOKMARK|Runaway argument|^! ' main.log 2>/dev/null | wc -l | tr -d ' ')
  [ "${undef}" = "0" ] && [ "${fatal}" = "0" ]
}

for i in 1 2 3 4; do
  latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1
  if clean_log; then break; fi
done

undef=$(grep -iE 'undefined (reference|citation)' main.log 2>/dev/null | wc -l | tr -d ' ')
fatal=$(grep -E 'BKM@entry|@@BOOKMARK|Runaway argument|^! ' main.log 2>/dev/null | wc -l | tr -d ' ')
pages=$(grep -o 'Output written on main.pdf ([0-9]* pages' main.log 2>/dev/null | tail -1 | grep -o '[0-9]*')

echo "------------------------------------------"
echo "Pasadas:                        ${i}"
echo "Referencias/citas sin resolver: ${undef}"
echo "Errores fatales:                ${fatal}"
echo "Páginas:                        ${pages:-(ver main.pdf)}"
if [ "${undef}" = "0" ] && [ "${fatal}" = "0" ]; then
  echo "BUILD OK"
else
  echo "BUILD CON AVISOS — revisar main.log"
  exit 1
fi
