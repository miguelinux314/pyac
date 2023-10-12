#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arithmetic Encoder based on the Nayuki project.

See:
    - [1] Nayuki Project https://github.com/nayuki/Reference-arithmetic-coding
    - [2] "Elements in information theory" by Cover and Thomas (especially Section 13.3)
    - [3] "JPEG 2000 Image compression fundamentals: standards and practice" by Taubman and Marcellin
      (especially Section 2.3)
"""
__author__ = "Miguel Hernández Cabronero <miguel.hernandez@uab.cat>"
__date__ = "11/10/2023"

from typing import List
from bitarray import bitarray


class ProbabilityTable:
    """For an alphabet with S symbols (s in {0, 1, ..., S-1}),
     represent the symbol probability masses P(s) and
     their cumulative distributions F(s) = sum(P(t), t < s).
     These are stored using P-bit integers P'(s) and F'(s),
     so that P(s) ~= P'(s)/2^P, F(s) ~= F'(s)/2^P, which
     introduces an inefficiency decreasing with P.
     """

    def __init__(
            self,
            probabilities: List[float],
            bit_precision: int,
            tolerance: float = 1e-10):
        """
        Initialize the table of probabilities.
        :param probabilities: list of S non-negative values with sum approximately 1
          that represent the probability mass of each symbol
        :param bit_precision: bit precision used to store probability masses
          and cumulative probabilities
        :param tolerance: maximum error tolerance for the sum of `probabilities`.
          If exceeded, a ValueError is raised.
        """
        self.bit_precision = bit_precision
        self.max_val = (1 << bit_precision) - 1

        # Floating-point probability masses P(s)
        self.f_probabilities = probabilities
        float_probability_error = abs(1 - sum(self.f_probabilities))
        if float_probability_error > tolerance:
            raise ValueError(f"Probability sum error ({fpe}) exceeds "
                             f"the tolerance ({tolerance})")

        # Floating-point cumulative probabily masses F(s)
        cumulative_distribution = 0
        self.f_cumulative_probabilities = []
        for p in self.f_probabilities:
            self.f_cumulative_probabilities.append(cumulative_distribution)
            cumulative_distribution += p
        self.f_cumulative_probabilities.append(1.0)

        # Integer probability masses P'(s)
        self.i_probabilities = [int((self.max_val * p))
                                for p in self.f_probabilities]
        self.i_probabilities[-1] += \
            (((1 << self.bit_precision) - 1) - sum(self.i_probabilities))

        # Integer cumulative probabily masses F'(s)
        self.i_cumulative_probabilities = [
            int(self.max_val * F)
            for F in self.f_cumulative_probabilities]
        self.i_cumulative_probabilities[-1] = self.max_val

    @property
    def symbol_count(self) -> int:
        return len(self.f_probabilities)


class ArithmeticBase:
    """Base class of both ArithmeticEncoder and ArithmeticDecoder.
    It provides the [low, high) range update function, common
    to both subclasses.
    """

    def __init__(
            self,
            bit_precision: int):
        """
        :param bit_precision: Bit precision for the interval
          extrema [low, high). It corresponds to N+P in [3],
          and N is fixed to 2 in this implementation.
        """
        # Precision constants
        self.interval_bit_precision = bit_precision
        if bit_precision <= 8:
            raise ValueError(f"Invalid {bit_precision=} <= 8")
        self.msb_to_lsb_shift = bit_precision - 1

        # Bit masks used during the update process
        self.first_msb = 1 << (self.interval_bit_precision - 1)
        self.second_msb = 1 << (self.interval_bit_precision - 2)
        self.all_ones = (1 << self.interval_bit_precision) - 1

        # Probability range, scaled by 2^bit_precision.
        # Following [2], A_n = self.high - self.low + 1.
        # Note that low is conceptually extended by infinite zeros,
        # and high by infinite ones.
        self.low = 0
        self.high = self.all_ones

    def get_probability_table(self, probabilities: List[float]) -> ProbabilityTable:
        """Return a ProbabilityTable instance given the list of
        (floating point) probability masses and the selected
        interval bit depth.
        """
        return ProbabilityTable(
            probabilities=probabilities,
            bit_precision=self.interval_bit_precision - 3)

    def update_interval(self, symbol: int, probability_table: ProbabilityTable):
        """Update the probability interval given the next symbol (in 0,...,S-1,
        where S is the number of symbols) and the table of probabilities.
        """
        # Update the probability interval
        range_length = self.high - self.low + 1

        self.low, self.high = (
            # Equivalent to c_{n+1} <- c_n + a_n*F(s) (see [2]).
            # Note that a_n is approximated by floor((hi-lo+1)/2^P),
            # where P is the probability precision
            self.low +
            ((probability_table.i_cumulative_probabilities[symbol] * range_length)
             >> probability_table.bit_precision),

            self.low +
            ((probability_table.i_cumulative_probabilities[symbol + 1] * range_length)
             >> probability_table.bit_precision) - 1)

        # When both the low and high ends of the interval
        # Share their MSB, it's time for renormalization.
        # In the encoder, the value of the MSB is writen,
        # along with any carry (underflow) bits.
        # In the decoder, the MSB is processed and the next
        # LSB is appended.
        while ((self.low ^ self.high) & self.first_msb) == 0:
            self.renormalize()
            self.low = (self.low << 1) & self.all_ones
            self.high = ((self.high << 1) & self.all_ones) | 1

        # While self.low = 01XXX...X and self.high = 10XXX...X,
        # the interval length is less than 2^(B-2), but low and high
        # do not share their MSB. The following loop calculates
        # the maximum number r such that
        # - at least r ones follow the 0 MSB in self.low
        # - at least r zeros follow the 1 MSB in self.high
        # and removes (shifts out) those r bits from both self.low and self.high,
        # without modifying their MSB.
        # The removed bits (ones in self.low and zeros in self.high)
        # are all identically affected by the carry operation
        # when computing the bit range self.high - self.low.
        # Note that, by design the MSB of self.high is always 1 and
        # the LSB of self.low is always 0
        while (~self.high & self.low & self.second_msb) != 0:
            self.carry_bit()
            self.low = (self.low << 1) ^ self.first_msb
            self.high = ((self.high << 1) & self.all_ones) | self.first_msb | 1

    def renormalize(self):
        """Method called by ArithmeticBase.update_interval when the need for
        renormalization is detected. This call is performed
        before modifying self.low or self.hight, which is done by ArithmeticBase.
        """
        raise NotImplementedError("Must be separately implemented "
                                  "in the Encoder and the Decoder")

    def carry_bit(self):
        """Method called by ArithmeticBase.update_interval when
        self.low = 01XX...XX and self.high = 10XX...XX is detected.
        This method is called before the second MSB is removed from
        both.
        """
        raise NotImplementedError("Must be separately implemented "
                                  "in the Encoder and the Decoder")


class ArithmeticEncoder(ArithmeticBase):
    def __init__(self, output_path: str, bit_precision: int = 32):
        """
        :param output_path: Path where the compressed data are stored
        :param bit_precision: number of bits used to store the scaled
           interval ranges. Warning: values too small will result
           in efficiency losses, and values too large in computational
           overhead.
        """
        super().__init__(bit_precision=bit_precision)
        self.carry_count = 0
        self.output_path = output_path
        self.output_file = open(self.output_path, "wb")
        self.bitout = bitarray()
        # Simple aliasing for more inteligible calls
        self.code_symbol = self.update_interval

    def code_symbol(
            self,
            symbol: int,
            probabilities: ProbabilityTable) -> None:
        """Code the next symbol s in {0, 1, ..., S-1}
        using the provided probability table for S symbols.
        :param symbol: symbol to be encoded
        :param probabilities: table of probabilities of the different
          symbols
        """
        raise RuntimeError("This should have been substituted "
                           "with self.update_range in __init__")

    def renormalize(self):
        """Save the MSB and any carry (underflow) bits
        to the output bitarray.
        """
        msb = self.low >> self.msb_to_lsb_shift
        self.bitout.append(msb)
        self.bitout.extend((msb ^ 1,) * self.carry_count)

        self.carry_count = 0

    def carry_bit(self):
        """Increase the counter of carry bits.
        """
        self.carry_count += 1

    def finish(self):
        if self.bitout is None:
            # Finish has been called more than once, ignore
            return
        # Write a last 1 to allow proper decoding
        self.bitout.append(1)
        # Flush bits to file and close everything
        self.bitout.tofile(self.output_file)
        self.output_file.close()
        self.bitout = None
        self.output_file = None


class ArithmeticDecoder(ArithmeticBase):
    def __init__(self, bitin: bitarray, bit_precision: int = 32):
        super().__init__(bit_precision=bit_precision)
        self.bitin = bitin

        # Read the first bits of code
        self.code = 0
        for _ in range(self.interval_bit_precision):
            self.code = (self.code << 1) | self.bitin.pop(0)

        # Constants specific to the decoder
        self.ones_except_msb = self.all_ones >> 1

    def decode_symbol(self, probability_table: ProbabilityTable) -> int:
        # Calculate the approximate probability of the next symbol
        # based on the current range and the current code (decoded bits)
        range_length = self.high - self.low + 1
        offset = self.code - self.low
        next_symbol_prob = ((offset + 1) * probability_table.max_val - 1) // range_length

        # Find the largest symbol whose cumulative distribution is lower than
        # the approximate symbol probability.
        # NOTE: This is a simple linear search with complexity O(N), it can be
        # improved with a binary search algorithm O(log N)
        next_symbol = 1
        while next_symbol < probability_table.symbol_count:
            # probability_table.f_cumulative_probabilities[next_symbol] <= next_symbol_prob:
            if probability_table.i_cumulative_probabilities[next_symbol] > next_symbol_prob:
                next_symbol -= 1
                break
            next_symbol += 1
        else:
            next_symbol = probability_table.symbol_count - 1
        assert 0 <= next_symbol < probability_table.symbol_count

        if next_symbol < probability_table.symbol_count - 1:
            # Update the interval based on the decoded symbol
            self.update_interval(symbol=next_symbol, probability_table=probability_table)

        return next_symbol

    def renormalize(self):
        """Shift out the MSB and read the LSB from the file.
        """
        self.code = ((self.code << 1) & self.all_ones) | self.get_next_bit()

    def carry_bit(self):
        """Replicate the coder's carry (underflow) procedure, where the
        second MSB is removed. Then read the LSB from file.
        """
        self.code = ((self.code & self.first_msb)
                     | ((self.code << 1) & self.ones_except_msb)
                     | self.get_next_bit())

    def get_next_bit(self):
        """Read and return the next input bit from file, or return 0 when the EOF is found.
        """
        try:
            return self.bitin.pop(0)
        except IndexError:
            return 0
