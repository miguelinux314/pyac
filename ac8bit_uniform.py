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
    uniform_probabilities = [1 / 256] * 256
    uniform_probabilities.append(2 ** -24)
    uniform_probabilities = [p / sum(uniform_probabilities) for p in uniform_probabilities]

    if options.action in ("c", "compress"):
        ae = ArithmeticEncoder(output_path=options.output_path,
                               symbol_count=257,
                               initial_probabilities=uniform_probabilities)
        for i, symbol in enumerate(open(options.input_path, "rb").read()):
            ae.code_symbol(symbol=symbol)
        # Add EOF and flush
        ae.code_symbol(symbol=len(uniform_probabilities) - 1)
        ae.finish()

    if options.action in ("d", "decompress"):
        with open(options.output_path, "wb") as f_reconstructed:
            ad = ArithmeticDecoder(input_path=options.input_path,
                                   symbol_count=257,
                                   initial_probabilities=uniform_probabilities)

            while True:
                next_symbol = ad.decode_symbol()
                if next_symbol == len(uniform_probabilities) - 1:
                    # Found EOF
                    break
                f_reconstructed.write(bytes((next_symbol,)))


if __name__ == '__main__':
    main()
