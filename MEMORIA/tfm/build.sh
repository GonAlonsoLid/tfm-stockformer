#!/usr/bin/env bash
# Compila la memoria del TFM de forma reproducible.
# biblatex (backend=biber) + hyperref pueden necesitar varias invocaciones en un
# árbol limpio para que las referencias cruzadas y el outline PDF converjan; por
# eso se repite latexmk hasta que el log queda limpio (máx. 4). Uso: ./build.sh
#
# Nota: main.log queda en latin-1 con líneas muy largas, así que grep lo trata
# como binario y devuelve conteos no fiables salvo que se fuerce modo texto (-a).
set -u
cd "$(dirname "$0")"
# Asegura que biber/latexmk de TinyTeX están en el PATH aunque se invoque desde
# un entorno no interactivo.
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"

clean_log() {
  local undef fatal
  undef=$(grep -aiE 'undefined (reference|citation)' main.log 2>/dev/null | wc -l | tr -d ' ')
  fatal=$(grep -aE 'BKM@entry|@@BOOKMARK|Runaway argument|^! ' main.log 2>/dev/null | wc -l | tr -d ' ')
  [ "${undef}" = "0" ] && [ "${fatal}" = "0" ]
}

for i in 1 2 3 4; do
  latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1
  if clean_log; then break; fi
done

undef=$(grep -aiE 'undefined (reference|citation)' main.log 2>/dev/null | wc -l | tr -d ' ')
fatal=$(grep -aE 'BKM@entry|@@BOOKMARK|Runaway argument|^! ' main.log 2>/dev/null | wc -l | tr -d ' ')
pages=$(grep -ao 'Output written on main.pdf ([0-9]* pages' main.log 2>/dev/null | tail -1 | grep -ao '[0-9]*')

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
