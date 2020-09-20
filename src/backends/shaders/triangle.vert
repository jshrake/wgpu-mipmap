#version 450
// https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
layout(location = 0) out vec2 v_uv;
void main() {
  v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(vec2(v_uv.x, v_uv.y) * 2.0 - 1.0, 0.0, 1.0);
  v_uv.y = 1.0 - v_uv.y;
}
