use super::{compute::*, copy::*, render::*};
use crate::core::*;

/// Generates mipmaps for textures with any usage using the compute, render, or copy backends.
#[derive(Debug)]
pub struct RecommendedMipmapGenerator {
    render: RenderMipmapGenerator,
    compute: ComputeMipmapGenerator,
}

/// A list of supported texture formats.
const SUPPORTED_FORMATS: [wgpu::TextureFormat; 17] = {
    use wgpu::TextureFormat;
    [
        TextureFormat::R8Unorm,
        TextureFormat::R8Snorm,
        TextureFormat::R16Float,
        TextureFormat::Rg8Unorm,
        TextureFormat::Rg8Snorm,
        TextureFormat::R32Float,
        TextureFormat::Rg16Float,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Snorm,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
        TextureFormat::Rg32Float,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgba32Float,
    ]
};

impl RecommendedMipmapGenerator {
    /// Creates a new `RecommendedMipmapGenerator`. Once created, it can be used repeatedly to
    /// generate mipmaps for any texture with a supported format.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::new_with_format_hints(device, &SUPPORTED_FORMATS)
    }

    /// Creates a new `RecommendedMipmapGenerator`. Once created, it can be used repeatedly to
    /// generate mipmaps for any texture with format specified in `format_hints`.
    pub fn new_with_format_hints(
        device: &wgpu::Device,
        format_hints: &[wgpu::TextureFormat],
    ) -> Self {
        for format in format_hints {
            if !SUPPORTED_FORMATS.contains(&format) {
                log::warn!("[RecommendedMipmapGenerator::new] No support for requested texture format {:?}", *format);
                log::warn!("[RecommendedMipmapGenerator::new] Attempting to continue, but calls to generate may fail or produce unexpected results.");
                continue;
            }
        }
        let render = RenderMipmapGenerator::new_with_format_hints(device, format_hints);
        let compute = ComputeMipmapGenerator::new_with_format_hints(device, format_hints);
        Self { render, compute }
    }
}

impl MipmapGenerator for RecommendedMipmapGenerator {
    fn generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        texture_descriptor: &wgpu::TextureDescriptor,
    ) -> Result<(), Error> {
        // compute backend
        match self
            .compute
            .generate(device, encoder, texture, texture_descriptor)
        {
            Err(e) => {
                log::debug!("[RecommendedMipmapGenerator::generate] compute error {}.\n falling back to render backend.", e);
            }
            ok => return ok,
        };
        // render backend
        match self
            .render
            .generate(device, encoder, texture, texture_descriptor)
        {
            Err(e) => {
                log::debug!("[RecommendedMipmapGenerator::generate] render error {}.\n falling back to copy backend.", e);
            }
            ok => return ok,
        };
        // copy backend
        match CopyMipmapGenerator::new(&self.render).generate(
            device,
            encoder,
            texture,
            texture_descriptor,
        ) {
            Err(e) => {
                log::debug!("[RecommendedMipmapGenerator::generate] copy error {}.", e);
            }
            ok => return ok,
        }
        Err(Error::UnsupportedUsage(texture_descriptor.usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[allow(dead_code)]
    async fn generate_and_copy_to_cpu_recommended(
        buffer: &[u8],
        texture_descriptor: &wgpu::TextureDescriptor<'_>,
    ) -> Result<Vec<MipBuffer>, Error> {
        let (_instance, _adaptor, device, queue) = wgpu_setup().await;
        let generator = crate::backends::RecommendedMipmapGenerator::new_with_format_hints(
            &device,
            &[texture_descriptor.format],
        );
        Ok(
            generate_and_copy_to_cpu(&device, &queue, &generator, buffer, texture_descriptor)
                .await?,
        )
    }
    #[test]
    fn checkerboard_r8_render() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_r8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::R8Unorm;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::RenderMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        dbg!(format);
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();

            assert!(mipmap_buffers.len() == mip_level_count as usize);

            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and the value is an average of 0 and 255
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                dbg!(data);
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                // The final result is implementation dependent
                // but we expect the pixel to be a perfect
                // blend of white and black, i.e 255 / 2 = 127.5
                // Depending on the platform and underlying implementation,
                // this might round up or down so check 127 and 128
                assert!(data[0] == 127 || data[0] == 128);
            }
        });
    }

    #[test]
    fn checkerboard_rgba8_render() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::RenderMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0 and 255, but in sRGB color space
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                assert!(data[0] == 127 || data[0] == 128);
                assert!(data[1] == 127 || data[1] == 128);
                assert!(data[2] == 127 || data[2] == 128);
                assert!(data[3] == 255);
            }
        });
    }

    #[test]
    fn checkerboard_srgba8_render() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::RenderMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0 and 255, but in sRGB color space
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                // The final result is implementation dependent
                // See https://entropymine.com/imageworsener/srgbformula/
                // for how to convert between linear and srgb
                // Where does 187 and 188 come from? Solve for x in:
                // ((((x / 255 + 0.055) / 1.055)^2.4) * 255) == (255 / 2)
                // -> x = 187.516155
                assert!(data[0] == 187 || data[0] == 188);
                assert!(data[1] == 187 || data[1] == 188);
                assert!(data[2] == 187 || data[2] == 188);
                assert!(data[3] == 255);
            }
        });
    }

    #[test]
    fn checkerboard_r8_compute() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_r8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::R8Unorm;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::ComputeMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();

            assert!(mipmap_buffers.len() == mip_level_count as usize);

            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and the value is an average of 0 and 255
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                // The final result is implementation dependent
                // but we expect the pixel to be a perfect
                // blend of white and black, i.e 255 / 2 = 127.5
                // Depending on the platform and underlying implementation,
                // this might round up or down so check 127 and 128
                assert!(data[0] == 127 || data[0] == 128);
            }
        });
    }

    #[test]
    fn checkerboard_rgba8_compute() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::ComputeMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0 and 255
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                assert!(data[0] == 127 || data[0] == 128);
                assert!(data[1] == 127 || data[1] == 128);
                assert!(data[2] == 127 || data[2] == 128);
                assert!(data[3] == 255);
            }
        });
    }

    #[test]
    fn checkerboard_srgba8_compute() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::ComputeMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0 and 255, but in sRGB color space
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                // The final result is implementation dependent
                // See https://entropymine.com/imageworsener/srgbformula/
                // for how to convert between linear and srgb
                // Where does 187 and 188 come from? Solve for x in:
                // ((((x / 255 + 0.055) / 1.055)^2.4) * 255) == (255 / 2)
                // -> x = 187.516155
                dbg!(data);
                assert!(data[0] == 187 || data[0] == 188);
                assert!(data[1] == 187 || data[1] == 188);
                assert!(data[2] == 187 || data[2] == 188);
                assert!(data[3] == 255);
            }
        });
    }

    #[test]
    fn checkerboard_rgba32f_render() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba32f(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba32Float;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::RenderMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(
                bytemuck::cast_slice(&data),
                &texture_descriptor,
            )
            .await
            .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0.0 and 1.0
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                let data: &[f32] = bytemuck::try_cast_slice(data).unwrap();
                dbg!(data);
                assert!((data[0] - 0.5).abs() < f32::EPSILON);
                assert!((data[1] - 0.5).abs() < f32::EPSILON);
                assert!((data[2] - 0.5).abs() < f32::EPSILON);
                assert!((data[3] - 1.0).abs() < f32::EPSILON);
            }
        });
    }

    #[test]
    fn checkerboard_rgba32f_compute() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba32f(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba32Float;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::ComputeMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(
                bytemuck::cast_slice(&data),
                &texture_descriptor,
            )
            .await
            .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0.0 and 1.0
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                let data: &[f32] = bytemuck::try_cast_slice(data).unwrap();
                dbg!(data);
                assert!((data[0] - 0.5).abs() < f32::EPSILON);
                assert!((data[1] - 0.5).abs() < f32::EPSILON);
                assert!((data[2] - 0.5).abs() < f32::EPSILON);
                assert!((data[3] - 1.0).abs() < f32::EPSILON);
            }
        });
    }

    #[test]
    fn checkerboard_rgba8_copy() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        let checkboard_size = 16;
        let data = checkerboard_rgba8(size, size, checkboard_size);
        // Create a texture
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            format,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: crate::CopyMipmapGenerator::required_usage()
                | wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST,
            label: None,
        };
        futures::executor::block_on(async {
            let mipmap_buffers = generate_and_copy_to_cpu_recommended(&data, &texture_descriptor)
                .await
                .unwrap();
            assert!(mipmap_buffers.len() == mip_level_count as usize);
            for mip in &mipmap_buffers {
                assert!(
                    mip.buffer.len()
                        == mip.dimensions.unpadded_bytes_per_row * mip.dimensions.height
                );
            }
            // The last mip map level should be 1x1 and each of the 4 components per pixel
            // should be the average of 0 and 255
            if let Some(mip) = mipmap_buffers.last() {
                let width = mip.dimensions.width;
                let height = mip.dimensions.height;
                let bpp = mip.dimensions.bytes_per_channel;
                let data = &mip.buffer;
                dbg!(data);
                assert!(width == 1);
                assert!(height == 1);
                assert!(data.len() == width * height * bpp);
                assert!(data[0] == 127 || data[0] == 128);
                assert!(data[1] == 127 || data[1] == 128);
                assert!(data[2] == 127 || data[2] == 128);
                assert!(data[3] == 255);
            }
        });
    }
}
