# wgpu-mipmap

![ci](https://github.com/jshrake/wgpu-mipmap/workflows/ci/badge.svg)

Generate mipmaps for [wgpu](https://github.com/gfx-rs/wgpu-rs) textures.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
wgpu-mipmap = "0.1"
```

Example usage:

```rust
use wgpu_mipmap::*;
fn example(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), Error> {
    // create a recommended generator
    let generator = RecommendedMipmapGenerator::new(&device);
    // create and upload data to a texture
    let texture_descriptor = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth: 1,
        },
        mip_level_count: 10, // 1 + log2(512)
        sample_count: 1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        dimension: wgpu::TextureDimension::D2,
        usage: wgpu::TextureUsage::STORAGE,
        label: None,
    };
    let texture = device.create_texture(&texture_descriptor);
    // upload_data_to_texture(&texture);
    // create an encoder and generate mipmaps for the texture
    let mut encoder = device.create_command_encoder(&Default::default());
    generator.generate(&device, &mut encoder, &texture, &texture_descriptor)?;
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
```

## Features

wgpu-mipmap is in the early stages of development and can only generate mipmaps for
2D textures with floating-point formats. The library implements several backends
in order to support various texture usage patterns:

- `ComputeMipmapGenerator`: For power of two textures with with usage
  `TextureUsage::STORAGE`. Uses a compute pipeline to generate mipmaps.
- `RenderMipmapGenerator`: For textures with usage
  `TextureUsage::OUTPUT_ATTACHMENT`. Uses a render pipeline to generate mipmaps.
- `CopyMipmapGenerator`: For textures with usage `TextureUsage::SAMPLED`.
  Allocates a new texture, uses a render pipeline to generate mipmaps in the new
  texture, then copies the result back to the original texture.
- `RecommendedMipmapGenerator`: Uses one of the above implementations depending
  on texture usage (prefers the compute backend, followed by the render backend,
  and finally the copy backend).

## Development

### Run the examples

The examples test various use cases and generate images for manual inspection and comparsion.

```console
$ cargo run --example cat
$ cargo run --example checkerboard
```

### Run the tests

```console
$ cargo test
```

### How to compile the shaders

```console
$ make build-shaders
```

See [src/shaders/README.md](src/shaders/README.md) for dependencies and more information.

## Benchmarks

TODO

## Resources

- https://github.com/gpuweb/gpuweb/issues/386
- https://github.com/gpuweb/gpuweb/issues/513
- https://github.com/gfx-rs/wgpu-rs/blob/master/examples/mipmap/main.rs
- https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBlitImage.html#_description
