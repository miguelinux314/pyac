#!/bin/bash

input=sample_data.raw
for code in ./ac8bit_uniform.py ./ac3symbol_uniform.py; do
   echo "Testing $code...."
   rm -f out.ac rec.raw; echo Compressing...; $code c $input out.ac && echo Decompressing... && $code d out.ac rec.raw && diff -qrsda $input rec.raw 
done
