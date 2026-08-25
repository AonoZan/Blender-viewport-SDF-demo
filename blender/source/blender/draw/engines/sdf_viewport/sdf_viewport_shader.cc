#include "sdf_viewport_shader.hh"
#include "GPU_shader.hh"
#include "intern/gpu_shader_create_info.hh"

namespace blender::sdf_viewport {

gpu::Shader *sdf_viewport_sdf_shader_create()
{
  static gpu::shader::ShaderCreateInfo *info = nullptr;
  
  if (!info) {
    info = new gpu::shader::ShaderCreateInfo("sdf_viewport_sdf");

    /* Declare dynamic fragment depth write to disable early-Z testing */
    info->depth_write(gpu::shader::DepthWrite::ANY);

    /* 1. Texture Binding */
    info->sampler(0, gpu::shader::ImageType::Float2D, "shapeDataTexture");

    /* 2. Vertex Input Layout */
    info->vertex_in(0, gpu::shader::Type::float2_t, "position");

    /* 3. Stage Interface */
    gpu::shader::StageInterfaceInfo *s_info = new gpu::shader::StageInterfaceInfo("v_ndc_interface", "");
    s_info->smooth(gpu::shader::Type::float2_t, "v_ndc");
    info->vertex_out(*s_info);

    /* 4. Output bindings */
    info->fragment_out(0, gpu::shader::Type::float4_t, "FragColor");

    /* 5. Uniform Parameters / Push Constants */
    info->push_constant(gpu::shader::Type::float4x4_t, "viewProjectionMatrix");
    info->push_constant(gpu::shader::Type::float4x4_t, "invViewProjectionMatrix");
    info->push_constant(gpu::shader::Type::float3_t, "cameraPos_world");
    info->push_constant(gpu::shader::Type::float3_t, "cameraForward_world");
    info->push_constant(gpu::shader::Type::float2_t, "viewportSize");
    info->push_constant(gpu::shader::Type::int_t, "isPerspective");
    info->push_constant(gpu::shader::Type::int_t, "numActiveShapes");
    info->push_constant(gpu::shader::Type::float_t, "maxDist");
    info->push_constant(gpu::shader::Type::float3_t, "debugColorStart");
    info->push_constant(gpu::shader::Type::float3_t, "debugColorEnd");

    /* 6. GLSL Source Bindings */
    info->vertex_source("sdf_viewport_sdf_vert.glsl");
    info->fragment_source("sdf_viewport_abuffer_frag.glsl");
  }

  return GPU_shader_create_from_info(
      reinterpret_cast<const GPUShaderCreateInfo *>(info)
  );
}

}  // namespace blender::sdf_viewport