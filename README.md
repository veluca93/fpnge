# Fast PNG Encoder
This is a proof-of-concept fast PNG encoder that uses AVX2 and a special
Huffman table to encode images faster. Speed on a single core is anywhere from
180 to 800 MP/s on a Threadripper 3970x, depending on compile time settings and
content.

At the moment, only RGB(A) input is supported.
