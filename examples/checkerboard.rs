#[cfg(feature = "debug")]
use renderdoc::{RenderDoc, V110};
use wgpu_mipmap::RecommendedMipmapGenerator;

fn main() {
    env_logger::init();
    #[cfg(feature = "debug")]
    let mut rd: RenderDoc<V110> = RenderDoc::new().expect("Unable to connect");
    #[cfg(feature = "debug")]
    rd.start_frame_capture(std::ptr::null(), std::ptr::null());
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    futures::executor::block_on((|| {
        async {
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
            // Generate texture data on the CPU
            let width = 512;
            let height = 512;
            let mip_level_count = 1 + (width.max(height) as f32).log2().floor() as u32;
            let data = wgpu_mipmap::util::checkerboard_rgba8(width, height, 16);
            let texture_extent = wgpu::Extent3d {
                width,
                height,
                depth: 1,
            };
            // Generate different mipmaps for both a linear and srgb format
            // with both the render and compute code paths
            let formats: std::collections::HashMap<_, _> = vec![
                ("linear", wgpu::TextureFormat::Rgba8Unorm),
                ("srgb", wgpu::TextureFormat::Rgba8UnormSrgb),
            ]
            .into_iter()
            .collect();
            let supported_usage: std::collections::HashMap<_, _> = vec![
                (
                    "compute",
                    wgpu_mipmap::ComputeMipmapGenerator::required_usage(),
                ),
                (
                    "render",
                    wgpu_mipmap::RenderMipmapGenerator::required_usage(),
                ),
                ("copy", wgpu_mipmap::CopyMipmapGenerator::required_usage()),
            ]
            .into_iter()
            .collect();
            let generator = RecommendedMipmapGenerator::new(&device);
            for (format_str, format) in &formats {
                for (usage_str, usage) in &supported_usage {
                    let texture_descriptor = wgpu::TextureDescriptor {
                        size: texture_extent,
                        mip_level_count,
                        format: *format,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        usage: *usage | wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::COPY_SRC,
                        label: None,
                    };

                    let mipmap_buffers = wgpu_mipmap::util::generate_and_copy_to_cpu(
                        &device,
                        &queue,
                        &generator,
                        &data,
                        &texture_descriptor,
                    )
                    .await
                    .expect("shouldn't fail");

                    let has_file_system_available = cfg!(not(target_arch = "wasm32"));
                    if !has_file_system_available {
                        return;
                    }

                    // Write the different mip levels as files
                    for (i, mip) in mipmap_buffers.iter().enumerate() {
                        image::save_buffer(
                            format!("checkerboard-{}-{}-{}.png", format_str, usage_str, i),
                            &mip.buffer,
                            mip.dimensions.width as u32,
                            mip.dimensions.height as u32,
                            image::ColorType::Rgba8,
                        )
                        .unwrap();
                    }
                }
            }
        }
    })());
    #[cfg(feature = "debug")]
    rd.end_frame_capture(std::ptr::null(), std::ptr::null());
}
