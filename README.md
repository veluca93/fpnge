# Fast PNG Encoder
This is a proof-of-concept fast PNG encoder that uses AVX2 and a special
Huffman table to encode images faster. Speed on a single core is anywhere from
180 to 800 MP/s on a Threadripper 3970x, depending on compile time settings and
content.

It supports 8 and 16 bit content, 1 to 4 channels; it can also emit
[cICP chunks](https://www.w3.org/TR/png/#cICP-chunk) for signaling that
the content should be interpreted as HDR.
