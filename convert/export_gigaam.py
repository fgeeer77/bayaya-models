#!/usr/bin/env python3
"""Exports a GigaAM CTC model to the sherpa-onnx NeMo CTC format, the same way the official
sherpa-onnx script does for GigaAM v3 (scripts/nemo/GigaAM/export-onnx-ctc-v3.py).

    python3 export_gigaam.py multilingual_ctc      # writes model.int8.onnx and tokens.txt
"""
import os
import sys

import gigaam
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def add_meta_data(filename: str, meta_data: dict):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    # Protobuf cannot hold more than 2 GB: the fp32 Large model keeps its weights in a side file.
    if model.ByteSize() >= LIMIT:
        onnx.save(model, filename, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(filename) + ".data")
    else:
        onnx.save(model, filename)


LIMIT = 2_000_000_000


def main():
    model_name = sys.argv[1]
    model = gigaam.load_model(model_name)
    print(model.cfg)
    vocabulary = model.cfg["decoding"]["vocabulary"]
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {len(vocabulary)}\n")
    model.to_onnx(".")
    add_meta_data(f"./{model_name}.onnx", {
        "vocab_size": len(vocabulary) + 1,
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecCTCModel",
        "version": "1",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
        "language": "Multilingual",
        "comment": model_name,
        "is_giga_am": 1,
    })
    large = os.path.exists(f"./{model_name}.onnx.data")
    output = "./model.int8.tmp.onnx" if large else "./model.int8.onnx"
    quantize_dynamic(model_input=f"./{model_name}.onnx", model_output=output, weight_type=QuantType.QUInt8,
                     use_external_data_format=large)
    if large:
        # The INT8 model is about a quarter of the size: store it as the single file sherpa-onnx loads.
        quantized = onnx.load(output)
        assert quantized.ByteSize() < LIMIT, quantized.ByteSize()
        onnx.save(quantized, "./model.int8.onnx")
    print("model.int8.onnx", os.path.getsize("./model.int8.onnx"), "bytes")


if __name__ == "__main__":
    main()
