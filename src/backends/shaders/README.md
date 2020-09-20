# Shaders

## Dependencies

In order to compile GLSL to SPIRV, the following binaries must be on your `$PATH`:

- `glslc` from [Google/shaderc](https://github.com/google/shaderc)
- `spirv-opt` from [KhronosGroup/SPIRV-Tools][https://github.com/khronosgroup/spirv-tools]

Then, you can run:

```console
$ ./compile.sh
```

This script handles generating all the compute shader combinations required by the code.
