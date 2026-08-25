#include "sdf_viewport_engine.hh"
#include "sdf_viewport_shader.hh"

#include "RE_engine.h"
#include "BLI_utildefines.h"
#include "BLI_string.h"
#include "BLT_translation.hh" 
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_view3d_types.h"
#include "DNA_mesh_types.h"
#include "DNA_material_types.h"
#include "BKE_idprop.hh"
#include "BKE_lib_id.hh"
#include "BKE_context.hh"
#include "BKE_layer.hh"
#include "BKE_material.hh"
#include "BKE_mesh.hh"
#include "BKE_mesh_runtime.hh"
#include "ED_view3d.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

/* Math libraries */
#include "BLI_math_vector.h"
#include "BLI_math_vector.hh"
#include "BLI_math_matrix.h"

#include "GPU_framebuffer.hh"
#include "GPU_texture.hh"
#include "GPU_immediate.hh"
#include "GPU_shader.hh"
#include "GPU_matrix.hh"
#include "GPU_state.hh"

#include "../../intern/draw_handle.hh"
#include "../../intern/draw_view.hh"
#include "../../intern/draw_view_data.hh"

namespace blender::sdf_viewport {

static Engine draw_engine_instance;

static gpu::Shader *sdf_shader = nullptr;
static gpu::Texture *shape_texture = nullptr;

constexpr int MAX_TEXTURE_SHAPES = 64;
constexpr int SHAPE_TEXTURE_WIDTH = 6;

DrawEngine *Engine::create_instance() {
  return new SdfViewportStorage();
}

void Engine::free_static() {
  if (sdf_shader) {
    GPU_shader_free(sdf_shader);
    sdf_shader = nullptr;
  }
  if (shape_texture) {
    GPU_texture_free(shape_texture);
    shape_texture = nullptr;
  }
}

void SdfViewportStorage::init() {
  active_shapes_count = 0;
  standard_objects.clear();
  sdf_shapes_data.resize(MAX_TEXTURE_SHAPES * SHAPE_TEXTURE_WIDTH * 4, 0.0f);
}

void SdfViewportStorage::begin_sync() {
  active_shapes_count = 0;
  standard_objects.clear();
  std::fill(sdf_shapes_data.begin(), sdf_shapes_data.end(), 0.0f);
}

bool SdfViewportStorage::is_sdf_object(Object *ob) {
  Object *orig_ob = ob->id.orig_id ? (Object *)ob->id.orig_id : ob;
  if (BLI_strncasecmp(orig_ob->id.name + 2, "SDF_Shape_", 10) == 0) {
    return true;
  }
  if (orig_ob->id.properties && IDP_GetPropertyFromGroup(orig_ob->id.properties, "sdf_type")) {
    return true;
  }
  return false;
}

void SdfViewportStorage::object_sync(draw::ObjectRef &ob_ref, draw::Manager &/*manager*/) {
  Object *ob = ob_ref.object;
  if (!ob) return;
  sync_object_manual(ob);
}

void SdfViewportStorage::sync_object_manual(Object *ob) {
  if (is_sdf_object(ob)) {
    if (active_shapes_count >= MAX_TEXTURE_SHAPES) {
      return;
    }

    Object *orig_ob = ob->id.orig_id ? (Object *)ob->id.orig_id : ob;
    IDProperty *properties = orig_ob->id.properties;
    if (!properties) return;

    IDProperty *type_prop = IDP_GetPropertyFromGroup(properties, "sdf_type");
    IDProperty *params_prop = IDP_GetPropertyFromGroup(properties, "sdf_params");
    if (!type_prop || !params_prop) return;

    float type_id = 0.0f;
    if (type_prop->type == IDP_STRING) {
      const char *type_str = (const char *)type_prop->data.pointer;
      if (type_str) {
        if (BLI_strcasecmp(type_str, "sphere") == 0)        type_id = 0.0f;
        else if (BLI_strcasecmp(type_str, "box") == 0)      type_id = 1.0f;
        else if (BLI_strcasecmp(type_str, "torus") == 0)    type_id = 2.0f;
        else if (BLI_strcasecmp(type_str, "cylinder") == 0) type_id = 3.0f;
      }
    } else if (type_prop->type == IDP_INT) {
      type_id = (float)type_prop->data.val;
    } else if (type_prop->type == IDP_FLOAT) {
      type_id = *(float *)&type_prop->data.val;
    } else if (type_prop->type == IDP_DOUBLE) {
      type_id = (float)*(double *)&type_prop->data.val;
    }

    float params[3] = {1.0f, 1.0f, 1.0f}; 
    if (params_prop->type == IDP_ARRAY && params_prop->data.pointer) {
      int subtype = params_prop->subtype;
      if (subtype == IDP_FLOAT) {
        float *arr = (float *)params_prop->data.pointer; 
        for (int p = 0; p < 3 && p < params_prop->len; ++p) params[p] = arr[p];
      } else if (subtype == IDP_DOUBLE) {
        double *arr = (double *)params_prop->data.pointer; 
        for (int p = 0; p < 3 && p < params_prop->len; ++p) params[p] = (float)arr[p];
      } else if (subtype == IDP_INT) {
        int *arr = (int *)params_prop->data.pointer; 
        for (int p = 0; p < 3 && p < params_prop->len; ++p) params[p] = (float)arr[p];
      }
    } else if (params_prop->type == IDP_FLOAT) {
      params[0] = *(float *)&params_prop->data.val;
    } else if (params_prop->type == IDP_DOUBLE) {
      params[0] = (float)*(double *)&params_prop->data.val;
    } else if (params_prop->type == IDP_INT) {
      params[0] = (float)params_prop->data.val;
    }

    int base_idx_params = (active_shapes_count * SHAPE_TEXTURE_WIDTH * 4);
    sdf_shapes_data[base_idx_params + 0] = type_id;
    for (int p = 0; p < 3; ++p) {
      sdf_shapes_data[base_idx_params + 1 + p] = params[p];
    }

    float obmat[4][4];
    copy_m4_m4(obmat, ob->object_to_world().ptr()); 

    for (int col = 0; col < 4; ++col) {
      int base_idx_col = base_idx_params + ((1 + col) * 4);
      for (int row = 0; row < 4; ++row) {
        sdf_shapes_data[base_idx_col + row] = obmat[col][row]; 
      }
    }

    int base_idx_sel = base_idx_params + (5 * 4);
    float sel_status = 0.0f;
    if ((ob->base_flag & BASE_SELECTED) || (orig_ob->base_flag & BASE_SELECTED)) {
      bool is_act = (active_object == ob || active_object == orig_ob);
      sel_status = is_act ? 2.0f : 1.0f;
    }
    sdf_shapes_data[base_idx_sel + 0] = sel_status;
    sdf_shapes_data[base_idx_sel + 1] = static_cast<float>(active_shapes_count + 1);
    sdf_shapes_data[base_idx_sel + 2] = 0.0f;
    sdf_shapes_data[base_idx_sel + 3] = 0.0f;

    active_shapes_count++;
  } else {
    if (ob->type == OB_MESH && ob->data) {
      standard_objects.push_back(ob);
    }
  }
}

static float3 matcap_color(const float3 &world_normal) {
  float3 light_dir = math::normalize(float3(0.5f, 0.5f, 1.0f));
  float light = std::clamp(math::dot(world_normal, light_dir), 0.0f, 1.0f);

  auto smoothstep = [](float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
  };

  float3 base_color(0.8f, 0.85f, 0.9f);
  float3 highlight(1.0f, 1.0f, 1.0f);
  float3 rim(0.3f, 0.3f, 0.35f);

  float t1 = smoothstep(0.0f, 0.5f, light);
  float3 color = math::interpolate(rim, base_color, t1);

  float t2 = smoothstep(0.7f, 1.0f, light);
  color = math::interpolate(color, highlight, t2);

  return color;
}

static void draw_standard_scene_objects(SdfViewportStorage *storage) {
  if (storage->standard_objects.empty()) {
    return;
  }

  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_depth_mask(true);

  GPUVertFormat *format = immVertexFormat();
  uint pos_attr = GPU_vertformat_attr_add(format, "pos", gpu::VertAttrType::SFLOAT_32_32_32);
  uint col_attr = GPU_vertformat_attr_add(format, "color", gpu::VertAttrType::SFLOAT_32_32_32_32);

  immBindBuiltinProgram(GPU_SHADER_3D_SMOOTH_COLOR);

  for (Object *ob : storage->standard_objects) {
    const Mesh *mesh = reinterpret_cast<const Mesh *>(ob->data);
    if (!mesh || mesh->verts_num == 0) {
      continue;
    }

    Span<float3> positions = mesh->vert_positions();
    if (positions.is_empty()) {
      continue;
    }

    Span<int3> corner_tris = mesh->corner_tris();
    Span<int> corner_verts = mesh->corner_verts();

    if (corner_tris.is_empty()) {
      continue;
    }

    float4x4 ob_mat = ob->object_to_world();

    GPU_matrix_push();
    GPU_matrix_mul(ob_mat.ptr());

    const size_t max_tris_per_batch = 2048;
    for (size_t i = 0; i < corner_tris.size(); i += max_tris_per_batch) {
      size_t tris_count = std::min(max_tris_per_batch, corner_tris.size() - i);

      immBegin(GPU_PRIM_TRIS, tris_count * 3);
      for (size_t j = 0; j < tris_count; ++j) {
        const int3 &tri = corner_tris[i + j];
        const float3 &p0 = positions[corner_verts[tri[0]]];
        const float3 &p1 = positions[corner_verts[tri[1]]];
        const float3 &p2 = positions[corner_verts[tri[2]]];

        float3 w0 = math::transform_point(ob_mat, p0);
        float3 w1 = math::transform_point(ob_mat, p1);
        float3 w2 = math::transform_point(ob_mat, p2);

        float3 world_facenor = math::cross(w1 - w0, w2 - w0);
        float nor_len = math::length(world_facenor);
        if (nor_len > 1e-6f) {
          world_facenor /= nor_len;
        } else {
          world_facenor = float3(0.0f, 0.0f, 1.0f);
        }

        float3 shaded_col = matcap_color(world_facenor);

        float c[4] = {
            shaded_col.x,
            shaded_col.y,
            shaded_col.z,
            1.0f
        };

        immAttr4fv(col_attr, c);
        immVertex3fv(pos_attr, p0);
        immAttr4fv(col_attr, c);
        immVertex3fv(pos_attr, p1);
        immAttr4fv(col_attr, c);
        immVertex3fv(pos_attr, p2);
      }
      immEnd();
    }

    GPU_matrix_pop();
  }

  immUnbindProgram();
}

static void draw_sdf_viewport_scene(gpu::FrameBuffer *active_fb, 
                                    SdfViewportStorage *storage,
                                    const float view_proj[4][4],
                                    const float inv_view_proj[4][4],
                                    const float cam_pos[3],
                                    const float cam_forward[3],
                                    bool is_persp) {
  if (!active_fb || !storage) {
    return;
  }

  /* 1. Draw Standard Geometry (writes color + depth) */
  draw_standard_scene_objects(storage);

  if (storage->active_shapes_count == 0) {
    return;
  }

  /* 2. Update SDF Data Texture */
  if (!shape_texture || GPU_texture_width(shape_texture) != SHAPE_TEXTURE_WIDTH) {
    if (shape_texture) {
      GPU_texture_free(shape_texture);
    }
    shape_texture = GPU_texture_create_2d(
        "ff_shape_texture", SHAPE_TEXTURE_WIDTH, MAX_TEXTURE_SHAPES, 1, 
        gpu::TextureFormat::SFLOAT_32_32_32_32, 
        GPU_TEXTURE_USAGE_SHADER_READ, nullptr 
    );
  }

  if (!shape_texture) return;
  GPU_texture_update(shape_texture, GPU_DATA_FLOAT, storage->sdf_shapes_data.data());

  if (!sdf_shader) {
    sdf_shader = sdf_viewport_sdf_shader_create();
    if (!sdf_shader) return;
  }

  int vp[4] = {0, 0, 1, 1};
  GPU_framebuffer_viewport_get(active_fb, vp);
  float viewport_size[2] = {
      static_cast<float>(vp[2] > 0 ? vp[2] : 1),
      static_cast<float>(vp[3] > 0 ? vp[3] : 1)
  };

  float max_dist = 200.0f;
  float debug_color_start[3] = {1.0f, 0.0f, 0.0f};
  float debug_color_end[3]   = {0.0f, 0.0f, 1.0f};

  GPU_matrix_push();
  GPU_matrix_push_projection();

  float identity[4][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
  };
  GPU_matrix_identity_set();
  GPU_matrix_projection_set(identity);

  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_depth_mask(true);

  immBindShader(sdf_shader);
  GPU_texture_bind(shape_texture, 0);

  GPU_shader_uniform_mat4(sdf_shader, "viewProjectionMatrix", view_proj);
  GPU_shader_uniform_mat4(sdf_shader, "invViewProjectionMatrix", inv_view_proj);
  GPU_shader_uniform_3fv(sdf_shader, "cameraPos_world", cam_pos);
  GPU_shader_uniform_3fv(sdf_shader, "cameraForward_world", cam_forward);
  GPU_shader_uniform_2fv(sdf_shader, "viewportSize", viewport_size);
  GPU_shader_uniform_1i(sdf_shader, "isPerspective", is_persp ? 1 : 0);
  GPU_shader_uniform_1i(sdf_shader, "numActiveShapes", storage->active_shapes_count);
  GPU_shader_uniform_1f(sdf_shader, "maxDist", max_dist);
  GPU_shader_uniform_3fv(sdf_shader, "debugColorStart", debug_color_start);
  GPU_shader_uniform_3fv(sdf_shader, "debugColorEnd", debug_color_end);

  /* Draw Fullscreen Quad */
  GPUVertFormat *quad_format = immVertexFormat();
  uint quad_pos_attr = GPU_vertformat_attr_add(quad_format, "position", gpu::VertAttrType::SFLOAT_32_32);

  immBegin(GPU_PRIM_TRIS, 6);
  immVertex2f(quad_pos_attr, -1.0f, -1.0f);
  immVertex2f(quad_pos_attr,  1.0f, -1.0f);
  immVertex2f(quad_pos_attr, -1.0f,  1.0f);
  immVertex2f(quad_pos_attr, -1.0f,  1.0f);
  immVertex2f(quad_pos_attr,  1.0f, -1.0f);
  immVertex2f(quad_pos_attr,  1.0f,  1.0f);
  immEnd();

  immUnbindProgram();
  GPU_texture_unbind(shape_texture);

  GPU_matrix_pop_projection();
  GPU_matrix_pop();

  /* Restore depth state for subsequent overlay/grid rendering */
  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_depth_mask(true);
}

void SdfViewportStorage::draw(draw::Manager &/*manager*/) {
}

static void sdf_viewport_view_update(RenderEngine * /*engine*/,
                                     const bContext * /*context*/,
                                     Depsgraph * /*depsgraph*/) {
}

static void sdf_viewport_view_draw(RenderEngine * /*engine*/,
                                   const bContext *context,
                                   Depsgraph *depsgraph) {
  gpu::FrameBuffer *active_fb = GPU_framebuffer_active_get();

  SdfViewportStorage *storage = static_cast<SdfViewportStorage *>(draw_engine_instance.create_instance());
  storage->init();
  if (context) {
    ViewLayer *view_layer = CTX_data_view_layer(context);
    if (view_layer) {
      storage->active_object = BKE_view_layer_active_object_get(view_layer);
    }
  }
  storage->begin_sync();

  DEGObjectIterSettings deg_iter_settings{};
  deg_iter_settings.depsgraph = depsgraph;
  deg_iter_settings.flags = DEG_ITER_OBJECT_FLAG_LINKED_DIRECTLY | DEG_ITER_OBJECT_FLAG_VISIBLE;

  DEG_OBJECT_ITER_BEGIN (&deg_iter_settings, ob) {
    storage->sync_object_manual(ob);
  } DEG_OBJECT_ITER_END;

  RegionView3D *rv3d = CTX_wm_region_view3d(context);
  ARegion *region = CTX_wm_region(context);

  float view_mat[4][4], proj_mat[4][4], view_proj[4][4], inv_view_proj[4][4], inv_view[4][4];
  float cam_pos[3], cam_forward[3];
  bool is_persp = true;

  if (rv3d && region) {
    copy_m4_m4(view_mat, rv3d->viewmat);
    copy_m4_m4(proj_mat, rv3d->winmat);
    copy_m4_m4(view_proj, rv3d->persmat);
    copy_m4_m4(inv_view_proj, rv3d->persinv);
    copy_m4_m4(inv_view, rv3d->viewinv);
    is_persp = (rv3d->winmat[3][3] == 0.0f);

    copy_v3_v3(cam_pos, rv3d->viewinv[3]);
    cam_forward[0] = -rv3d->viewinv[2][0];
    cam_forward[1] = -rv3d->viewinv[2][1];
    cam_forward[2] = -rv3d->viewinv[2][2];
  } else {
    GPU_matrix_model_view_get(view_mat);
    GPU_matrix_projection_get(proj_mat);
    mul_m4_m4m4(view_proj, proj_mat, view_mat);
    invert_m4_m4(inv_view_proj, view_proj);
    invert_m4_m4(inv_view, view_mat);
    is_persp = (proj_mat[3][3] == 0.0f);

    cam_pos[0] = inv_view[3][0];
    cam_pos[1] = inv_view[3][1];
    cam_pos[2] = inv_view[3][2];
    cam_forward[0] = -inv_view[2][0];
    cam_forward[1] = -inv_view[2][1];
    cam_forward[2] = -inv_view[2][2];
  }
  normalize_v3(cam_forward);

  GPU_matrix_push();
  GPU_matrix_push_projection();
  GPU_matrix_set(view_mat);
  GPU_matrix_projection_set(proj_mat);

  draw_sdf_viewport_scene(active_fb, storage, view_proj, inv_view_proj, cam_pos, cam_forward, is_persp);

  GPU_matrix_pop_projection();
  GPU_matrix_pop();

  delete storage;
}

}  // namespace blender::sdf_viewport

namespace blender {
RenderEngineType DRW_engine_viewport_sdf_viewport_type = {
    nullptr,
    nullptr,
    "SDF_VIEWPORT",
    N_("SDF Viewport"),
    blender::RE_USE_PREVIEW | blender::RE_USE_STEREO_VIEWPORT | blender::RE_USE_GPU_CONTEXT,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    blender::sdf_viewport::sdf_viewport_view_update,
    blender::sdf_viewport::sdf_viewport_view_draw,
    nullptr,
    nullptr,
    nullptr,
    (blender::DrawEngineType *)&blender::sdf_viewport::draw_engine_instance,
    { nullptr, nullptr, nullptr },
};
}  // namespace blender