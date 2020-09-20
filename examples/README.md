# examples

The examples write out the generated mipmaps as pngs to the root directory for visual inspection.

## cat

Generates mipmaps for an sRGB png [cat.png](cat.png) ([https://commons.wikimedia.org/wiki/File:Avatar_cat.png](https://commons.wikimedia.org/wiki/File:Avatar_cat.png)) using the various backends.

```console
$ cargo run --example cat
```

## checkerboard

Generates mipmaps for both a linear and srgb checkerboard texture with the various backends.

```console
$ cargo run --example checkerboard
```
