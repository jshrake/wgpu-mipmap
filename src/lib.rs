/*!
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
*/
mod backends;
mod core;

#[doc(hidden)]
pub mod util;

#[doc(inline)]
pub use crate::backends::{
    ComputeMipmapGenerator, CopyMipmapGenerator, RecommendedMipmapGenerator, RenderMipmapGenerator,
};

#[doc(inline)]
pub use crate::core::*;
