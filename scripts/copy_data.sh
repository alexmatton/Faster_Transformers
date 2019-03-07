#!/usr/bin/env bash

INPUT_DIR='../datasets/chunked'
OUTPUT_DIR='../datasets/cnn_full'
DEBUG_DIR='../datasets/cnn_debug'

#takes chunks from input_dir and puts them in output_dir train

mkdir -p ${OUTPUT_DIR}/train
mkdir -p ${OUTPUT_DIR}/val
mkdir -p ${OUTPUT_DIR}/test

mkdir -p ${DEBUG_DIR}/train
mkdir -p ${DEBUG_DIR}/val
mkdir -p ${DEBUG_DIR}/test

cp ${INPUT_DIR}/train* ${OUTPUT_DIR}/train/
cp ${INPUT_DIR}/val* ${OUTPUT_DIR}/val/
cp ${INPUT_DIR}/test* ${OUTPUT_DIR}/test/

# creating debug datasets
cp ${INPUT_DIR}/train_000.bin ${DEBUG_DIR}/train/
#cp ${INPUT_DIR}/train_001.bin ${DEBUG_DIR}/train/
#cp ${INPUT_DIR}/train_002.bin ${DEBUG_DIR}/train/
#cp ${INPUT_DIR}/train_003.bin ${DEBUG_DIR}/train/
#cp ${INPUT_DIR}/train_004.bin ${DEBUG_DIR}/train/

cp ${INPUT_DIR}/train_000.bin ${DEBUG_DIR}/val/val_000.bin


cp ${INPUT_DIR}/train_000.bin ${DEBUG_DIR}/test/test_000.bin

