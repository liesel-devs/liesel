#!/usr/bin/env sh

PATTERN="1s/^/# GENERATED FILE, DO NOT EDIT!\n\n/
         s/import numpy/import jax.numpy/g
         s/from numpy/from jax.numpy/g
         s/import scipy/import jax.scipy/g
         s/from scipy/from jax.scipy/g
         s/substrates.numpy/substrates.jax/g
         s/tf2numpy/tf2jax/g"

if [ ! -d "$1/numpy" ]; then
    echo "$1/numpy not found"
    exit 1
fi

rm -r "$1/jax"
rsync -am --include="*/" --include="*.py" --exclude="*" "$1/numpy/" "$1/jax/"
find "$1/jax" -name "*.py" -exec sed -i "$PATTERN" {} +
