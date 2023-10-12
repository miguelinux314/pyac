# pyAC - Python Arithmetic Codec

Basic prototype of an Arithmetic Coder and Decoder. Designed to:

- Help understand the underlying finite-precision arithmetic
- Practice and analyze static and adaptive probability models

Have a lot of fun!

-- Miguel Hernández-Cabronero https://deic.uab.cat/~mhernandez

## Dependencies

`pip install bitarray`

## Contents

* pyac.py: main library module. Includes the ArithmeticEncoder, ArithmeticDecoder and ProbabilityTable classes. 
* ac8bit_uniform.py: example 8-bit codec implementation using a static, uniform probability model.
    To be used as a template (don't expect any compression!)

## References

- [1] Nayuki Project https://github.com/nayuki/Reference-arithmetic-coding
- [2] "Elements in information theory" by Cover and Thomas (especially Section 13.3)
- [3] "JPEG 2000 Image compression fundamentals: standards and practice" by Taubman and Marcellin
  (especially Section 2.3)
