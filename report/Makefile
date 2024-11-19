FILE=report

all: pdf

pdf:
	pdflatex -shell-escape $(FILE).tex

clean:
	rm -f $(FILE).aux $(FILE).log $(FILE).out $(FILE).toc $(FILE).nav $(FILE).snm $(FILE).vrb $(FILE).pyg $(FILE).fls $(FILE).fdb_latexmk

.PHONY: all pdf clean
