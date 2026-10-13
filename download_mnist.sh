#!/bin/sh
# Descarcă cele 4 fișiere MNIST în data/
set -e
mkdir -p data
cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "$f" ]; then
        wget -q "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
        gunzip "$f.gz"
    fi
done
echo "Gata: fisierele MNIST sunt in data/"
