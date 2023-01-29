md := $(patsubst %.qmd, ../md/%.md, $(wildcard *.qmd))

.PHONY: render clean

render: $(md)

../md/%.md: %.qmd _quarto.yml
	mkdir -p ../md
	cp $^ ../md
	quarto render ../md/$<
	-rm -r $(addprefix ../md/, $^) ../md/.quarto

clean:
	-rm ../md/*.md
	-rm -r ../md/*_cache
	-rm -r ../md/*_files
