#version 450
// Expected that the sampler has mag_filter set to linear
layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(location = 0) out vec4 out_color;
layout(location = 0) in vec2 v_uv;

void main() {
  out_color = textureLod(sampler2D(u_texture, u_sampler), v_uv, 0.0);
}
