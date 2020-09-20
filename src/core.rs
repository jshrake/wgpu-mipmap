use thiserror::Error;

/// MipmapGenerator describes types that can generate mipmaps for a texture.
pub trait MipmapGenerator {
    /// Encodes commands to generate mipmaps for a texture.
    ///
    /// Expectations:
    /// - `texture_descriptor` should be the same descriptor used to create the `texture`.
    fn generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        texture_descriptor: &wgpu::TextureDescriptor,
    ) -> Result<(), Error>;
}

/// An error that occurred during mipmap generation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("Unsupported texture usage `{0:?}`.\nYour texture usage must contain one of: 1. TextureUsage::STORAGE, 2. TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED, 3. TextureUsage::COPY_SRC | TextureUsage::COPY_DST")]
    UnsupportedUsage(wgpu::TextureUsage),
    #[error(
        "Unsupported texture dimension `{0:?}. You texture dimension must be TextureDimension::D2`"
    )]
    UnsupportedDimension(wgpu::TextureDimension),
    #[error("Unsupported texture format `{0:?}`. Try using the render backend.")]
    UnsupportedFormat(wgpu::TextureFormat),
    #[error("Unsupported texture size. Texture size must be a power of 2.")]
    NpotTexture,
    #[error("Unknown texture format `{0:?}`.\nDid you mean to specify it in `MipmapGeneratorDescriptor::formats`?")]
    UnknownFormat(wgpu::TextureFormat),
}
