#!/bin/bash

echo "compiling for N=16"
aoc -o lcr_10_16.aocx -v ../src/lcr.cl -D M=10 -D N=16 --board pcie385n_d5
aoc -o lcr_30_16.aocx -v ../src/lcr.cl -D M=30 -D N=16 --board pcie385n_d5
aoc -o lcr_90_16.aocx -v ../src/lcr.cl -D M=90 -D N=16 --board pcie385n_d5
aoc -o lcr_180_16.aocx -v ../src/lcr.cl -D M=180 -D N=16 --board pcie385n_d5
aoc -o lcr_300_16.aocx -v ../src/lcr.cl -D M=300 -D N=16 --board pcie385n_d5

echo "compiling for N=9"
aoc -o lcr_10_9.aocx -v ../src/lcr.cl -D M=10 -D N=9 --board pcie385n_d5
aoc -o lcr_30_9.aocx -v ../src/lcr.cl -D M=30 -D N=9 --board pcie385n_d5
aoc -o lcr_90_9.aocx -v ../src/lcr.cl -D M=90 -D N=9 --board pcie385n_d5
aoc -o lcr_180_9.aocx -v ../src/lcr.cl -D M=180 -D N=9 --board pcie385n_d5
aoc -o lcr_300_9.aocx -v ../src/lcr.cl -D M=300 -D N=9 --board pcie385n_d5

echo "compiling for N=25"
aoc -o lcr_10_25.aocx -v ../src/lcr.cl -D M=10 -D N=25 --board pcie385n_d5
aoc -o lcr_30_25.aocx -v ../src/lcr.cl -D M=30 -D N=25 --board pcie385n_d5
aoc -o lcr_90_25.aocx -v ../src/lcr.cl -D M=90 -D N=25 --board pcie385n_d5
aoc -o lcr_180_25.aocx -v ../src/lcr.cl -D M=180 -D N=25 --board pcie385n_d5
aoc -o lcr_300_25.aocx -v ../src/lcr.cl -D M=300 -D N=25 --board pcie385n_d5

echo "compiling for N=36"
aoc -o lcr_10_36.aocx -v ../src/lcr.cl -D M=10 -D N=36 --board pcie385n_d5
aoc -o lcr_30_36.aocx -v ../src/lcr.cl -D M=30 -D N=36 --board pcie385n_d5
aoc -o lcr_90_36.aocx -v ../src/lcr.cl -D M=90 -D N=36 --board pcie385n_d5
aoc -o lcr_180_36.aocx -v ../src/lcr.cl -D M=180 -D N=36 --board pcie385n_d5
aoc -o lcr_300_36.aocx -v ../src/lcr.cl -D M=300 -D N=36 --board pcie385n_d5

echo "compiling for N=64"
aoc -o lcr_10_64.aocx -v ../src/lcr.cl -D M=10 -D N=64 --board pcie385n_d5
aoc -o lcr_30_64.aocx -v ../src/lcr.cl -D M=30 -D N=64 --board pcie385n_d5
aoc -o lcr_90_64.aocx -v ../src/lcr.cl -D M=90 -D N=64 --board pcie385n_d5
aoc -o lcr_180_64.aocx -v ../src/lcr.cl -D M=180 -D N=64 --board pcie385n_d5
aoc -o lcr_300_64.aocx -v ../src/lcr.cl -D M=300 -D N=64 --board pcie385n_d5