// sdf_viewport_shader.hh
#pragma once

namespace blender::gpu { class Shader; }

namespace blender::sdf_viewport {

gpu::Shader *sdf_viewport_sdf_shader_create();

}  // namespace blender::sdf_viewport