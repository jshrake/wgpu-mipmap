#!/usr/bin/env bash
set -e

function compile {
  glslc -c $1 -o $2
  spirv-opt -Os $2 -o $2
}


cd "$(dirname "$0")"
compile triangle.vert  triangle.vert.spv
compile box.frag box.frag.spv
compile box_srgb.comp box_srgb.comp.spv
compile box_srgb_macos.comp box_srgb_macos.comp.spv

# https://www.khronos.org/opengl/wiki/Image_Load_Store#Format_qualifiers
SUPPORTED_FORMATS=(
  rgba32f
  rgba16f
  rg32f
  rg16f
  r32f
  r16f
  rgba8
  rg8
  r8
  r11f_g11f_b10f
  rgb10_a2
  rgba16_snorm
  rgba8_snorm
  rg16_snorm
  rg8_snorm
  r16_snorm
  r8_snorm
)
for FORMAT in ${SUPPORTED_FORMATS[@]}; do
  (FORMAT=${FORMAT} envsubst < box.comp) > box_${FORMAT}.comp
  compile box_${FORMAT}.comp box_${FORMAT}.comp.spv
  rm box_${FORMAT}.comp
done
