/// utilities used throughout the project. Not part of the official API.
use crate::core::*;

#[derive(Debug)]
pub struct MipBuffer {
    pub buffer: Vec<u8>,
    pub dimensions: MipBufferDimensions,
    pub level: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct MipBufferDimensions {
    pub width: usize,
    pub height: usize,
    pub bytes_per_channel: usize,
    pub unpadded_bytes_per_row: usize,
    pub padded_bytes_per_row: usize,
}

impl MipBufferDimensions {
    pub fn new(width: usize, height: usize, bytes_per_channel: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let unpadded_bytes_per_row = width * bytes_per_channel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            bytes_per_channel,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

pub async fn generate_and_copy_to_cpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    generator: &dyn MipmapGenerator,
    data: &[u8],
    texture_descriptor: &wgpu::TextureDescriptor<'_>,
) -> Result<Vec<MipBuffer>, Error> {
    // Create a texture
    let buffer_dimensions = MipBufferDimensions::new(
        texture_descriptor.size.width as usize,
        texture_descriptor.size.height as usize,
        format_bytes_per_channel(&texture_descriptor.format),
    );
    let texture = device.create_texture(&texture_descriptor);
    // Upload `data` to the texture
    queue.write_texture(
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &data,
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: buffer_dimensions.unpadded_bytes_per_row as u32,
            rows_per_image: 0,
        },
        wgpu::Extent3d {
            width: buffer_dimensions.width as u32,
            height: buffer_dimensions.height as u32,
            depth: 1,
        },
    );
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    generator.generate(&device, &mut encoder, &texture, &texture_descriptor)?;
    // Copy all mipmap levels, including the base, to GPU buffers
    let buffers = {
        let mut buffers = Vec::new();
        for i in 0..texture_descriptor.mip_level_count {
            let mip_width = buffer_dimensions.width / 2usize.pow(i);
            let mip_height = buffer_dimensions.height / 2usize.pow(i);
            let mip_dimensions = MipBufferDimensions::new(
                mip_width,
                mip_height,
                buffer_dimensions.bytes_per_channel,
            );
            let size = (mip_dimensions.height * mip_dimensions.padded_bytes_per_row) as u64;
            let mip_texture_extent = wgpu::Extent3d {
                width: mip_width as u32,
                height: mip_height as u32,
                depth: 1,
            };
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TextureCopyView {
                    texture: &texture,
                    mip_level: i,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::BufferCopyView {
                    buffer: &buffer,
                    layout: wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: mip_dimensions.padded_bytes_per_row as u32,
                        rows_per_image: 0,
                    },
                },
                mip_texture_extent,
            );
            buffers.push((buffer, mip_dimensions));
        }
        buffers
    };
    queue.submit(std::iter::once(encoder.finish()));
    // Copy the GPU buffers to the CPU
    let mut mip_buffers: Vec<MipBuffer> = Vec::new();
    for (level, (buffer, buffer_dimensions)) in buffers.iter().enumerate() {
        // Note that we're not calling `.await` here.
        let buffer_slice = buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        device.poll(wgpu::Maintain::Wait);
        match buffer_future.await {
            Err(e) => panic!("Unexpected failure: {}", e),
            Ok(()) => {
                let padded_buffer = buffer_slice.get_mapped_range();
                // The buffer we get back is padded, so only extract what we need
                let mut exact_buffer = Vec::with_capacity(
                    buffer_dimensions.unpadded_bytes_per_row * buffer_dimensions.height,
                );
                for y in 0..buffer_dimensions.height {
                    let row_beg = y * buffer_dimensions.padded_bytes_per_row;
                    let row_end = row_beg + buffer_dimensions.unpadded_bytes_per_row;
                    exact_buffer.extend_from_slice(&padded_buffer[row_beg..row_end]);
                }
                mip_buffers.push(MipBuffer {
                    buffer: exact_buffer,
                    dimensions: *buffer_dimensions,
                    level: level as u32,
                });
            }
        }
    }
    Ok(mip_buffers)
}

pub fn checkerboard_r8(width: u32, height: u32, n: u32) -> Vec<u8> {
    use std::iter;

    (0..width * height)
        .flat_map(|id| {
            let x = id % width;
            let y = id / height;
            let v = (((x / n + y / n) % 2) * 255) as u8;
            iter::once(v)
        })
        .collect()
}

#[doc(hidden)]
pub fn checkerboard_rgba8(width: u32, height: u32, n: u32) -> Vec<u8> {
    use std::iter;

    (0..width * height)
        .flat_map(|id| {
            let x = id % width;
            let y = id / height;
            let v = (((x / n + y / n) % 2) * 255) as u8;
            iter::once(v)
                .chain(iter::once(v))
                .chain(iter::once(v))
                .chain(iter::once(255))
        })
        .collect()
}

#[doc(hidden)]
pub fn checkerboard_rgba32f(width: u32, height: u32, n: u32) -> Vec<f32> {
    use std::iter;

    (0..width * height)
        .flat_map(|id| {
            let x = id % width;
            let y = id / height;
            let v = ((x / n + y / n) % 2) as f32;
            iter::once(v)
                .chain(iter::once(v))
                .chain(iter::once(v))
                .chain(iter::once(1.0))
        })
        .collect()
}

fn format_bytes_per_channel(format: &wgpu::TextureFormat) -> usize {
    use wgpu::TextureFormat;
    match format {
        // 8 bit per channel
        TextureFormat::R8Unorm => 1,
        TextureFormat::R8Snorm => 1,
        TextureFormat::R8Uint => 1,
        TextureFormat::R8Sint => 1,
        // 16 bit per channel
        TextureFormat::R16Uint => 2,
        TextureFormat::R16Sint => 2,
        TextureFormat::R16Float => 2,
        TextureFormat::Rg8Unorm => 2,
        TextureFormat::Rg8Snorm => 2,
        TextureFormat::Rg8Uint => 2,
        TextureFormat::Rg8Sint => 2,
        // 32 bit per channel
        TextureFormat::R32Uint => 4,
        TextureFormat::R32Sint => 4,
        TextureFormat::R32Float => 4,
        TextureFormat::Rg16Uint => 4,
        TextureFormat::Rg16Sint => 4,
        TextureFormat::Rg16Float => 4,
        TextureFormat::Rgba8Unorm => 4,
        TextureFormat::Rgba8Snorm => 4,
        TextureFormat::Rgba8Uint => 4,
        TextureFormat::Rgba8Sint => 4,
        TextureFormat::Bgra8Unorm => 4,
        TextureFormat::Bgra8UnormSrgb => 4,
        TextureFormat::Rgba8UnormSrgb => 4,
        // packed 32 bit per channel
        TextureFormat::Rgb10a2Unorm => 4,
        TextureFormat::Rg11b10Float => 4,
        // 64 bit per channel
        TextureFormat::Rg32Uint => 8,
        TextureFormat::Rg32Sint => 8,
        TextureFormat::Rg32Float => 8,
        TextureFormat::Rgba16Uint => 8,
        TextureFormat::Rgba16Sint => 8,
        TextureFormat::Rgba16Float => 8,
        // 128 bit per channel
        TextureFormat::Rgba32Uint => 16,
        TextureFormat::Rgba32Sint => 16,
        TextureFormat::Rgba32Float => 16,
        _ => unimplemented!(),
    }
}

#[doc(hidden)]
#[allow(dead_code)]
pub(crate) async fn wgpu_setup() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropiate adapter");
    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .expect("Failed to create device");
    (instance, adapter, device, queue)
}

#[doc(hidden)]
#[allow(dead_code)]
pub(crate) fn get_mip_extent(extent: &wgpu::Extent3d, level: u32) -> wgpu::Extent3d {
    let mip_width = ((extent.width as f32) / (2u32.pow(level) as f32)).floor() as u32;
    let mip_height = ((extent.height as f32) / (2u32.pow(level) as f32)).floor() as u32;
    let mip_depth = ((extent.depth as f32) / (2u32.pow(level) as f32)).floor() as u32;
    wgpu::Extent3d {
        width: mip_width.max(1),
        height: mip_height.max(1),
        depth: mip_depth.max(1),
    }
}
