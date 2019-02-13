#!/bin/bash
# A little script to deep copy files while retaining parent structure
# Change parent directory and other variables as needed

src=""
dst=""

subdir="deep"
parent="hell"

for X in ./test*
do
	dir="$X/$subdir"
	#src="$src $dir"
	dst="$parent/$X"

	mkdir -p "$dst"
	cp -r "$dir" "$dst"
done

