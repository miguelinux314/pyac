#!/usr/bin/env python3
"""Example AC that employs a static, uniform probability model.
"""
__author__ = "Miguel Hernández-Cabronero"
__since__ = "2023/10/12"

import argparse
from bitarray import bitarray
from pyac import ArithmeticEncoder, ArithmeticDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        choices=("compress", "c", "decompress", "d"))
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    options = parser.parse_args()

    # Uniform distribution for 8-bit symbols plus a low-probability EOF symbol
    probs = [1 / 256] * 256
    probs.append(2 ** -24)
    probs = [p / sum(probs) for p in probs]

    if options.action in ("c", "compress"):
        ae = ArithmeticEncoder(output_path=options.output_path)
        probability_table = ae.get_probability_table(probabilities=probs)
        for i, symbol in enumerate(open(options.input_path, "rb").read()):
            ae.code_symbol(symbol=symbol,
                           probability_table=probability_table)
        # Add EOF and flush
        ae.code_symbol(symbol=len(probs) - 1, probability_table=probability_table)
        ae.finish()

    if options.action in ("d", "decompress"):
        with open(options.input_path, "rb") as f_compressed, \
                open(options.output_path, "wb") as f_reconstructed:
            bitin = bitarray()
            bitin.fromfile(f_compressed)
            ad = ArithmeticDecoder(bitin=bitin)
            probability_table = ad.get_probability_table(probabilities=probs)

            while True:
                next_symbol = ad.decode_symbol(probability_table=probability_table)
                if next_symbol == len(probs) - 1:
                    # Found EOF
                    break
                f_reconstructed.write(bytes((next_symbol,)))


if __name__ == '__main__':
    main()
