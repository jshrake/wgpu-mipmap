use crate::core::*;
use crate::util::get_mip_extent;
use log::warn;
use std::collections::HashMap;

/// Generates mipmaps for textures with storage usage.
#[derive(Debug)]
pub struct ComputeMipmapGenerator {
    layout_cache: HashMap<wgpu::TextureFormat, wgpu::BindGroupLayout>,
    pipeline_cache: HashMap<wgpu::TextureFormat, wgpu::ComputePipeline>,
}

impl ComputeMipmapGenerator {
    /// Returns the texture usage `ComputeMipmapGenerator` requires for mipmap generation.
    pub fn required_usage() -> wgpu::TextureUsage {
        wgpu::TextureUsage::STORAGE
    }

    /// Creates a new `ComputeMipmapGenerator`. Once created, it can be used repeatedly to
    /// generate mipmaps for any texture with format specified in `format_hints`.
    pub fn new_with_format_hints(
        device: &wgpu::Device,
        format_hints: &[wgpu::TextureFormat],
    ) -> Self {
        let mut layout_cache = HashMap::new();
        let mut pipeline_cache = HashMap::new();
        for format in format_hints {
            if let Some(module) = shader_for_format(device, format) {
                let bind_group_layout = bind_group_layout_for_format(device, format);
                let pipeline =
                    compute_pipeline_for_format(device, &module, &bind_group_layout, format);
                layout_cache.insert(*format, bind_group_layout);
                pipeline_cache.insert(*format, pipeline);
            } else {
                warn!(
                    "ComputeMipmapGenerator does not support requested format {:?}",
                    format
                );
                continue;
            }
        }
        Self {
            layout_cache,
            pipeline_cache,
        }
    }
}

impl MipmapGenerator for ComputeMipmapGenerator {
    fn generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        texture_descriptor: &wgpu::TextureDescriptor,
    ) -> Result<(), Error> {
        // Texture width and height must be a power of 2
        if !texture_descriptor.size.width.is_power_of_two()
            || !texture_descriptor.size.height.is_power_of_two()
        {
            return Err(Error::NpotTexture);
        }
        // Texture dimension must be 2D
        if texture_descriptor.dimension != wgpu::TextureDimension::D2 {
            return Err(Error::UnsupportedDimension(texture_descriptor.dimension));
        }
        if !texture_descriptor.usage.contains(Self::required_usage()) {
            return Err(Error::UnsupportedUsage(texture_descriptor.usage));
        }

        let layout = self
            .layout_cache
            .get(&texture_descriptor.format)
            .ok_or(Error::UnknownFormat(texture_descriptor.format))?;
        let pipeline = self
            .pipeline_cache
            .get(&texture_descriptor.format)
            .ok_or(Error::UnknownFormat(texture_descriptor.format))?;

        let mip_count = texture_descriptor.mip_level_count;
        // TODO: Can we create the views every call?
        let views = (0..mip_count)
            .map(|base_mip_level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level,
                    level_count: std::num::NonZeroU32::new(1),
                    array_layer_count: None,
                    base_array_layer: 0,
                })
            })
            .collect::<Vec<_>>();
        // Now dispatch the compute pipeline for each mip level
        // TODO: Likely need more flexibility here
        // - The compute shaders must have matching local_size_x and local_size_y values
        // - When the image size is less than 32x32, more work is performed than required
        let x_work_group_count = 32;
        let y_work_group_count = 32;
        for mip in 1..mip_count as usize {
            let src_view = &views[mip - 1];
            let dst_view = &views[mip];
            let mip_ext = get_mip_extent(&texture_descriptor.size, mip as u32);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch(
                (mip_ext.width / x_work_group_count).max(1),
                (mip_ext.height / y_work_group_count).max(1),
                1,
            );
        }
        Ok(())
    }
}

fn shader_for_format(
    device: &wgpu::Device,
    format: &wgpu::TextureFormat,
) -> Option<wgpu::ShaderModule> {
    use wgpu::TextureFormat;
    let s = |d| Some(device.create_shader_module(wgpu::util::make_spirv(d)));
    match format {
        TextureFormat::R8Unorm => s(include_bytes!("shaders/box_r8.comp.spv")),
        TextureFormat::R8Snorm => s(include_bytes!("shaders/box_r8_snorm.comp.spv")),
        TextureFormat::R16Float => s(include_bytes!("shaders/box_r16f.comp.spv")),
        TextureFormat::Rg8Unorm => s(include_bytes!("shaders/box_rg8.comp.spv")),
        TextureFormat::Rg8Snorm => s(include_bytes!("shaders/box_rg8_snorm.comp.spv")),
        TextureFormat::R32Float => s(include_bytes!("shaders/box_r32f.comp.spv")),
        TextureFormat::Rg16Float => s(include_bytes!("shaders/box_rg16f.comp.spv")),
        TextureFormat::Rgba8Unorm => s(include_bytes!("shaders/box_rgba8.comp.spv")),
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb => {
            // On MacOS, my GPUFamily2 v1 capable GPU
            // seems to perform the srgb -> linear before I load it
            // in the shader, but expects me to perform the linear -> srgb
            // conversion before storing.
            #[cfg(target_os = "macos")]
            {
                s(include_bytes!("shaders/box_srgb_macos.comp.spv"))
            }
            // On  Vulkan (and DX12?), the implementation does not perform
            // any conversion, so this shader handles it all
            #[cfg(not(target_os = "macos"))]
            {
                s(include_bytes!("shaders/box_srgb.comp.spv"))
            }
        }
        TextureFormat::Rgba8Snorm => s(include_bytes!("shaders/box_rgba8_snorm.comp.spv")),
        TextureFormat::Bgra8Unorm => s(include_bytes!("shaders/box_rgba8.comp.spv")),
        TextureFormat::Rgb10a2Unorm => s(include_bytes!("shaders/box_rgb10_a2.comp.spv")),
        TextureFormat::Rg11b10Float => s(include_bytes!("shaders/box_r11f_g11f_b10f.comp.spv")),
        TextureFormat::Rg32Float => s(include_bytes!("shaders/box_rg32f.comp.spv")),
        TextureFormat::Rgba16Float => s(include_bytes!("shaders/box_rgba16f.comp.spv")),
        TextureFormat::Rgba32Float => s(include_bytes!("shaders/box_rgba32f.comp.spv")),
        _ => None,
    }
}

fn bind_group_layout_for_format(
    device: &wgpu::Device,
    format: &wgpu::TextureFormat,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    dimension: wgpu::TextureViewDimension::D2,
                    format: *format,
                    readonly: true,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    dimension: wgpu::TextureViewDimension::D2,
                    format: *format,
                    readonly: false,
                },
                count: None,
            },
        ],
    })
}

fn compute_pipeline_for_format(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    format: &wgpu::TextureFormat,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("wgpu-mipmap-compute-pipeline-{:?}", format)),
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &module,
            entry_point: "main",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[allow(dead_code)]
    async fn generate_and_copy_to_cpu_compute(
        buffer: &[u8],
        texture_descriptor: &wgpu::TextureDescriptor<'_>,
    ) -> Result<Vec<MipBuffer>, Error> {
        let (_instance, _adaptor, device, queue) = wgpu_setup().await;
        let generator = crate::backends::ComputeMipmapGenerator::new_with_format_hints(
            &device,
            &[texture_descriptor.format],
        );
        Ok(
            generate_and_copy_to_cpu(&device, &queue, &generator, buffer, texture_descriptor)
                .await?,
        )
    }

    async fn generate_test(texture_descriptor: &wgpu::TextureDescriptor<'_>) -> Result<(), Error> {
        let (_instance, _adapter, device, _queue) = wgpu_setup().await;
        let generator =
            ComputeMipmapGenerator::new_with_format_hints(&device, &[texture_descriptor.format]);
        let texture = device.create_texture(&texture_descriptor);
        let mut encoder = device.create_command_encoder(&Default::default());
        generator.generate(&device, &mut encoder, &texture, &texture_descriptor)
    }

    #[test]
    fn sanity_check() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
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
            usage: ComputeMipmapGenerator::required_usage(),
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_ok());
        })());
    }

    #[test]
    fn unsupported_npot() {
        init();
        // Generate texture data on the CPU
        let size = 511;
        let mip_level_count = 1 + (size as f32).log2() as u32;
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
            usage: ComputeMipmapGenerator::required_usage(),
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_err());
            assert!(res.err() == Some(Error::NpotTexture));
        })());
    }

    #[test]
    fn unsupported_usage() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
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
            usage: wgpu::TextureUsage::empty(),
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_err());
            assert!(res.err() == Some(Error::UnsupportedUsage(wgpu::TextureUsage::empty())));
        })());
    }

    #[test]
    fn unknown_format() {
        init();
        // Generate texture data on the CPU
        let size = 512;
        let mip_level_count = 1 + (size as f32).log2() as u32;
        // Create a texture
        let format = wgpu::TextureFormat::Rg16Sint;
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
            usage: ComputeMipmapGenerator::required_usage(),
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_err());
            assert!(res.err() == Some(Error::UnknownFormat(wgpu::TextureFormat::Rg16Sint)));
        })());
    }
}
