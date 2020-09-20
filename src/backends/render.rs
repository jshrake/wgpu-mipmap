use crate::core::*;
use crate::util::get_mip_extent;
use log::warn;
use std::collections::HashMap;

/// Generates mipmaps for textures with output attachment usage.
#[derive(Debug)]
pub struct RenderMipmapGenerator {
    sampler: wgpu::Sampler,
    layout_cache: HashMap<wgpu::TextureComponentType, wgpu::BindGroupLayout>,
    pipeline_cache: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl RenderMipmapGenerator {
    /// Returns the texture usage `RenderMipmapGenerator` requires for mipmap generation.
    pub fn required_usage() -> wgpu::TextureUsage {
        wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED
    }

    /// Creates a new `RenderMipmapGenerator`. Once created, it can be used repeatedly to
    /// generate mipmaps for any texture with format specified in `format_hints`.
    pub fn new_with_format_hints(
        device: &wgpu::Device,
        format_hints: &[wgpu::TextureFormat],
    ) -> Self {
        // A sampler for box filter with clamp to edge behavior
        // In practice, the final result may be implementation dependent
        // - [Vulkan](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#textures-texel-linear-filtering)
        // - [Metal](https://developer.apple.com/documentation/metal/mtlsamplerminmagfilter/linear)
        // - [DX12](https://docs.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_filter)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&"wgpu-mipmap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let render_layout_cache = {
            let mut layout_cache = HashMap::new();
            // For now, we only cache a bind group layout for floating-point textures
            for component_type in &[wgpu::TextureComponentType::Float] {
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("wgpu-mipmap-bg-layout-{:?}", component_type)),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStage::FRAGMENT,
                                ty: wgpu::BindingType::SampledTexture {
                                    dimension: wgpu::TextureViewDimension::D2,
                                    component_type: *component_type,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStage::FRAGMENT,
                                ty: wgpu::BindingType::Sampler { comparison: false },
                                count: None,
                            },
                        ],
                    });
                layout_cache.insert(*component_type, bind_group_layout);
            }
            layout_cache
        };

        let render_pipeline_cache = {
            let mut pipeline_cache = HashMap::new();
            let vertex_module = device.create_shader_module(wgpu::util::make_spirv(
                include_bytes!("shaders/triangle.vert.spv"),
            ));
            let box_filter = device.create_shader_module(wgpu::util::make_spirv(include_bytes!(
                "shaders/box.frag.spv"
            )));
            for format in format_hints {
                let fragment_module = &box_filter;

                let component_type = wgpu::TextureComponentType::from(*format);
                if let Some(bind_group_layout) = render_layout_cache.get(&component_type) {
                    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[bind_group_layout],
                        push_constant_ranges: &[],
                    });
                    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(&format!("wgpu-mipmap-render-pipeline-{:?}", format)),
                        layout: Some(&layout),
                        vertex_stage: wgpu::ProgrammableStageDescriptor {
                            module: &vertex_module,
                            entry_point: "main",
                        },
                        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                            module: &fragment_module,
                            entry_point: "main",
                        }),
                        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: wgpu::CullMode::Back,
                            ..Default::default()
                        }),
                        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                        color_states: &[(*format).into()],
                        depth_stencil_state: None,
                        vertex_state: wgpu::VertexStateDescriptor {
                            index_format: wgpu::IndexFormat::Uint16,
                            vertex_buffers: &[],
                        },
                        sample_count: 1,
                        sample_mask: !0,
                        alpha_to_coverage_enabled: false,
                    });
                    pipeline_cache.insert(*format, pipeline);
                } else {
                    warn!(
                        "RenderMipmapGenerator does not support requested format {:?}",
                        format
                    );
                    continue;
                }
            }
            pipeline_cache
        };

        Self {
            sampler,
            layout_cache: render_layout_cache,
            pipeline_cache: render_pipeline_cache,
        }
    }

    /// Generate mipmaps from level 0 of `src_texture` to
    /// levels `dst_mip_offset..dst_texture_descriptor.mip_level_count`
    // of `dst_texture`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_src_dst(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_texture: &wgpu::Texture,
        dst_texture: &wgpu::Texture,
        src_texture_descriptor: &wgpu::TextureDescriptor,
        dst_texture_descriptor: &wgpu::TextureDescriptor,
        dst_mip_offset: u32,
    ) -> Result<(), Error> {
        let src_format = src_texture_descriptor.format;
        let src_mip_count = src_texture_descriptor.mip_level_count;
        let src_ext = src_texture_descriptor.size;
        let src_dim = src_texture_descriptor.dimension;
        let src_usage = src_texture_descriptor.usage;
        let src_next_mip_ext = get_mip_extent(&src_ext, 1);

        let dst_format = dst_texture_descriptor.format;
        let dst_mip_count = dst_texture_descriptor.mip_level_count;
        let dst_ext = dst_texture_descriptor.size;
        let dst_dim = dst_texture_descriptor.dimension;
        let dst_usage = dst_texture_descriptor.usage;
        // invariants that we expect callers to uphold
        if src_format != dst_format {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture formats must be equal");
        }
        if src_dim != dst_dim {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture dimensions must be eqaul");
        }
        if !((src_mip_count == dst_mip_count && src_ext == dst_ext)
            || (src_next_mip_ext == dst_ext))
        {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture extents must match or dst must be half the size of src");
        }

        if src_dim != wgpu::TextureDimension::D2 {
            return Err(Error::UnsupportedDimension(src_dim));
        }
        // src texture must be sampled
        if !src_usage.contains(wgpu::TextureUsage::SAMPLED) {
            return Err(Error::UnsupportedUsage(src_usage));
        }
        // dst texture must be sampled and output attachment
        if !dst_usage.contains(Self::required_usage()) {
            return Err(Error::UnsupportedUsage(dst_usage));
        }
        let format = src_format;
        let pipeline = self
            .pipeline_cache
            .get(&format)
            .ok_or(Error::UnknownFormat(format))?;
        let component_type = wgpu::TextureComponentType::from(format);
        let layout = self
            .layout_cache
            .get(&component_type)
            .ok_or(Error::UnknownFormat(format))?;
        let views = (0..src_mip_count)
            .map(|mip_level| {
                // The first view is mip level 0 of the src texture
                // Subsequent views are for the dst_texture
                let (texture, base_mip_level) = if mip_level == 0 {
                    (src_texture, 0)
                } else {
                    (dst_texture, mip_level - dst_mip_offset)
                };
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
        for mip in 1..src_mip_count as usize {
            let src_view = &views[mip - 1];
            let dst_view = &views[mip];
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
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

impl MipmapGenerator for RenderMipmapGenerator {
    fn generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        texture_descriptor: &wgpu::TextureDescriptor,
    ) -> Result<(), Error> {
        self.generate_src_dst(
            device,
            encoder,
            &texture,
            &texture,
            &texture_descriptor,
            &texture_descriptor,
            0,
        )
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
    async fn generate_and_copy_to_cpu_render(
        buffer: &[u8],
        texture_descriptor: &wgpu::TextureDescriptor<'_>,
    ) -> Result<Vec<MipBuffer>, Error> {
        let (_instance, _adaptor, device, queue) = wgpu_setup().await;
        let generator = crate::backends::RenderMipmapGenerator::new_with_format_hints(
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
            RenderMipmapGenerator::new_with_format_hints(&device, &[texture_descriptor.format]);
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
            usage: RenderMipmapGenerator::required_usage(),
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_ok());
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
        let format = wgpu::TextureFormat::Rgba8Sint;
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
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            label: None,
        };
        futures::executor::block_on((|| async {
            let res = generate_test(&texture_descriptor).await;
            assert!(res.is_err());
            assert!(res.err() == Some(Error::UnknownFormat(wgpu::TextureFormat::Rgba8Sint)));
        })());
    }
}
