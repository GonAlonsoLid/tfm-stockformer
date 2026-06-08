# Configuración de latexmk para la memoria del TFM
# biblatex con backend=bibtex requiere ejecutar bibtex y varias pasadas.
$pdf_mode = 1;                       # pdflatex
$bibtex_use = 2;                     # ejecuta bibtex y limpia .bbl en -C
$max_repeat = 7;                     # margen para estabilizar refs cruzadas + biblatex
# nonstopmode (no -halt-on-error): sobrevive transitorios de hyperref/bookmark
# en la primera pasada de un arranque en frío; latexmk converge en pasadas sucesivas.
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$clean_ext = 'bbl run.xml bcf synctex.gz fdb_latexmk fls -blx.bib';
