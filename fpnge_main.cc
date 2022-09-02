// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <thread>
#include <vector>

#include "fpnge.h"
#include "lodepng.h"

static int print_usage(const char *app) {
  fprintf(stderr, "Usage: %s [options] in.png out.png\n", app);
  fprintf(stderr, "  -1..%d  compression level (default %d)\n",
          FPNGE_COMPRESS_LEVEL_BEST, FPNGE_COMPRESS_LEVEL_DEFAULT);
  fprintf(stderr, "  -r<n>  run <n> repetitions and report\n");
  fprintf(stderr, "  -pq    reinterpret input pixels as PQ and add a cICP "
                  "chunk; can be used to make HDR PNGs\n");
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    return print_usage(argv[0]);
  }

  int comp_level = FPNGE_COMPRESS_LEVEL_DEFAULT;
  size_t num_reps = 0;
  int cicp_colorspace = FPNGE_CICP_NONE;

  int arg_p = 1;
  for (; arg_p < argc; arg_p++) {
    if (argv[arg_p][0] != '-')
      break;
    char opt = argv[arg_p][1];
    if (opt >= '1' && opt <= '0' + FPNGE_COMPRESS_LEVEL_BEST) {
      comp_level = opt - '0';
    } else if (opt == 'r') {
      num_reps = atoi(argv[arg_p] + 2);
    } else if (opt == 'p' && argv[arg_p][2] == 'q') {
      cicp_colorspace = FPNGE_CICP_PQ;
    } else {
      return print_usage(argv[0]);
    }
  }

  if (arg_p + 2 != argc) {
    return print_usage(argv[0]);
  }

  const char *in = argv[arg_p];
  const char *out = argv[arg_p + 1];
  struct FPNGEOptions options;
  FPNGEFillOptions(&options, comp_level, cicp_colorspace);

  FILE *infile = fopen(in, "rb");
  if (!infile) {
    fprintf(stderr, "error opening %s: %s\n", in, strerror(errno));
    return 1;
  }
  fseek(infile, 0, SEEK_END);
  std::vector<unsigned char> in_data(ftell(infile));
  fseek(infile, 0, SEEK_SET);
  if (fread(in_data.data(), 1, in_data.size(), infile) != in_data.size()) {
    fprintf(stderr, "error reading from %s: %s\n", in, strerror(errno));
    exit(1);
  }

  LodePNGState state;
  lodepng_state_init(&state);

  bool has_alpha = false;
  bool is_hbd = false;

  unsigned width, height;
  unsigned error =
      lodepng_inspect(&width, &height, &state, in_data.data(), in_data.size());
  if (error) {
    fprintf(stderr, "lodepng error %u: %s\n", error, lodepng_error_text(error));
    return 1;
  }

  if (state.info_png.color.colortype == LCT_RGBA ||
      state.info_png.color.colortype == LCT_GREY_ALPHA) {
    has_alpha = true;
  }

  if (state.info_png.color.bitdepth > 8) {
    is_hbd = true;
  }

  if (lodepng_chunk_find_const(in_data.data() + 8,
                               in_data.data() + in_data.size(), "tRNS")) {
    has_alpha = true;
  }

  size_t num_c = has_alpha ? 4 : 3;

  unsigned char *png;

  // RGB(A) only for now.
  error = lodepng_decode_memory(
      &png, &width, &height, in_data.data(), in_data.size(),
      has_alpha ? LodePNGColorType::LCT_RGBA : LodePNGColorType::LCT_RGB,
      is_hbd ? 16 : 8);

  if (error) {
    fprintf(stderr, "lodepng error %u: %s\n", error, lodepng_error_text(error));
    return 1;
  }

  size_t encoded_size = 0;
  size_t bytes_per_channel = is_hbd ? 2 : 1;
  void *encoded =
      malloc(FPNGEOutputAllocSize(bytes_per_channel, num_c, width, height));

  if (num_reps > 0) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t _ = 0; _ < num_reps; _++) {
      encoded_size = FPNGEEncode(bytes_per_channel, num_c, png, width,
                                 width * num_c * bytes_per_channel, height,
                                 encoded, &options);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    size_t pixels = size_t{width} * size_t{height} * num_reps;
    float mps = pixels / us;
    fprintf(stderr, "%10.3f MP/s\n", mps);
    fprintf(stderr, "%10.3f bits/pixel\n",
            encoded_size * 8.0 / float(width) / float(height));
  } else {
    encoded_size = FPNGEEncode(bytes_per_channel, num_c, png, width,
                               width * num_c * bytes_per_channel, height,
                               encoded, &options);
  }

  FILE *o = fopen(out, "wb");
  if (!o) {
    fprintf(stderr, "error opening %s: %s\n", out, strerror(errno));
    return 1;
  }
  if (fwrite(encoded, 1, encoded_size, o) != encoded_size) {
    fprintf(stderr, "error writing to %s: %s\n", out, strerror(errno));
  }
  fclose(o);
  free(png);
  free(encoded);

  return 0;
}
