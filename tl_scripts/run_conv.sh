#!/bin/bash

shapes=(
    "1 2048 7 7 512 1 1 0 1 1"
    "1 512 14 14 512 3 3 1 2 1"
    "1 512 14 14 1024 1 1 0 1 1"
    "1 1024 14 14 256 1 1 0 1 1"
    "1 256 28 28 256 3 3 1 2 1"
    "1 256 28 28 512 1 1 0 1 1"
    "1 512 28 28 128 1 1 0 1 1"
    "1 128 56 56 256 1 1 0 1 1"
    "1 256 56 56 64 1 1 0 1 1"
    "1 64 56 56 64 3 3 1 1 1"
    "1 64 56 56 64 1 1 0 1 1"
    "1 64 56 56 256 1 1 0 1 1"
    "1 512 56 56 256 1 1 0 2 1"
    "1 128 28 28 128 3 3 1 1 1"
    "1 128 28 28 512 1 1 0 1 1"
    "1 1024 28 28 512 1 1 0 2 1"
    "1 256 14 14 256 3 3 1 1 1"
    "1 256 14 14 1024 1 1 0 1 1"
    "1 2048 14 14 1024 1 1 0 2 1"
    "1 512 7 7 512 3 3 1 1 1"
    "1 512 7 7 2048 1 1 0 1 1"
    "128 2048 7 7 512 1 1 0 1 1"
    "128 512 14 14 512 3 3 1 2 1"
    "128 512 14 14 1024 1 1 0 1 1"
    "128 1024 14 14 256 1 1 0 1 1"
    "128 256 28 28 256 3 3 1 2 1"
    "128 256 28 28 512 1 1 0 1 1"
    "128 512 28 28 128 1 1 0 1 1"
    "128 128 56 56 256 1 1 0 1 1"
    "128 256 56 56 64 1 1 0 1 1"
    "128 64 56 56 64 3 3 1 1 1"
    "128 64 56 56 64 1 1 0 1 1"
    "128 64 56 56 256 1 1 0 1 1"
    "128 512 56 56 256 1 1 0 2 1"
    "128 128 28 28 128 3 3 1 1 1"
    "128 128 28 28 512 1 1 0 1 1"
    "128 1024 28 28 512 1 1 0 2 1"
    "128 256 14 14 256 3 3 1 1 1"
    "128 256 14 14 1024 1 1 0 1 1"
    "128 2048 14 14 1024 1 1 0 2 1"
    "128 512 7 7 512 3 3 1 1 1"
    "128 512 7 7 2048 1 1 0 1 1"
)

mkdir -p logs/
id=0
for shape in "${shapes[@]}"; do
    read n c h w inc kw kh p s d <<< "$shape"

    python conv_tune.py \
        --n "$n" \
        --c "$c" \
        --h "$h" \
        --w "$w" \
        --inc "$inc" \
        --kw "$kw" \
        --kh "$kh" \
        --p "$p" \
        --s "$s" \
        --d "$d" \
        2>&1 | tee "logs/${id}.conv_${n}_${c}_${h}_${w}_${inc}_${kw}_${kh}_${p}_${s}_${d}.log"
    id=$((id + 1))
done