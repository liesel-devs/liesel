md := $(patsubst %.qmd, ../md/%.md, $(wildcard *.qmd))

.PHONY: render postrender clean

render: $(md) postrender

../md/_quarto.yml: _quarto.yml
	mkdir -p ../md
	cp $< $@

../md/%.qmd: %.qmd
	mkdir -p ../md
	cp $< $@

../md/%.md: ../md/%.qmd ../md/_quarto.yml
	quarto render $<

postrender:
	-rm ../md/_quarto.yml
	-rm -r ../md/.quarto

clean:
	-rm ../md/*.md
	-rm -r ../md/*_cache
	-rm -r ../md/*_files
