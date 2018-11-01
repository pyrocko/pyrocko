all: sinlookup generate run

clean:
	@rm -f *.o
	@rm -f *.s
	@rm -f *.i
	@rm -f sinlookup

sinlookup: lut.c sinlookup.c
	gcc -g -Wall -O3 -save-temps lut.c sinlookup.c -o sinlookup -lm
	@rm -f *.o
	@rm -f *.i

run: sinlookup
	@echo
	./sinlookup -r

generate: sinlookup
	@echo
	./sinlookup -g
