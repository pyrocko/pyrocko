#!/bin/bash

cd pyrocko/data/earthmodels
for f in *.f.nd ; do 
    cake simplify-model --model=$f --accuracy=0.001 >${f/%.f.nd/.m.nd} 
    cake simplify-model --model=$f --accuracy=0.002 >${f/%.f.nd/.l.nd} 
done

