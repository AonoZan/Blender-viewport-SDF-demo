#include "sdf_viewport_engine.hh"
#include "sdf_viewport_shader.hh"

#include "RE_engine.h"
#include "BLI_utildefines.h"
#include "BLI_string.h"
#include "BLT_translation.hh" 
#include "DNA_object_types.h"
#include "DNA_view3d_types.h"
#include "BKE_idprop.hh"
#include "BKE_lib_id.hh"
#include "BKE_context.hh"
#include "ED_view3d.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

/* Math libraries */
#include "BLI_math_vector.h"
#include "BLI_math_matrix.h"

#include "GPU_framebuffer.hh"
#include "GPU_texture.hh"
#include "GPU_immediate.hh"
#include "GPU_shader.hh"
#include "GPU_matrix.hh"

/* Required to fully define ObjectRef and avoid incomplete type error */
#include "../../intern/draw_handle.hh"
#include "../../intern/draw_view.hh"
#include "../../intern/draw_view_data.hh"
#include <iostream>
#include "GPU_state.hh"

namespace blender::sdf_viewport {

// Static C++ drawing engine registration instance
static Engine draw_engine_instance;

static gpu::Shader *sdf_shader = nullptr;
static gpu::Texture *shape_texture = nullptr;

constexpr int MAX_TEXTURE_SHAPES = 64;
constexpr int SHAPE_TEXTURE_WIDTH = 5; // 1 texel parameter + 4 texels matrix column

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
  sdf_shapes_data.resize(MAX_TEXTURE_SHAPES * SHAPE_TEXTURE_WIDTH * 4, 0.0f);
}

void SdfViewportStorage::begin_sync() {
  active_shapes_count = 0;
  std::fill(sdf_shapes_data.begin(), sdf_shapes_data.end(), 0.0f);
}

/* Sync shapes natively using the Draw Manager's object loop */
void SdfViewportStorage::object_sync(draw::ObjectRef &ob_ref, draw::Manager &/*manager*/) {
  Object *ob = ob_ref.object;
  if (!ob) return;
  sync_object_manual(ob);
}

void SdfViewportStorage::sync_object_manual(Object *ob) {
  if (active_shapes_count >= MAX_TEXTURE_SHAPES) {
    return;
  }

  Object *orig_ob = ob->id.orig_id ? (Object *)ob->id.orig_id : ob;

  if (BLI_strncasecmp(orig_ob->id.name + 2, "SDF_Shape_", 10) != 0) {
    return;
  }

  IDProperty *properties = orig_ob->id.properties;
  if (!properties) return;

  IDProperty *type_prop = IDP_GetPropertyFromGroup(properties, "sdf_type");
  IDProperty *params_prop = IDP_GetPropertyFromGroup(properties, "sdf_params");
  if (!type_prop || !params_prop) return;

  /* 1. Extract Shape Type */
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

  /* 2. Extract Parameters */
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

  /* 3. Extract Evaluated Object Transform Matrix */
  float obmat[4][4];
  copy_m4_m4(obmat, ob->object_to_world().ptr()); 

  /* Pack column-by-column */
  for (int col = 0; col < 4; ++col) {
    int base_idx_col = base_idx_params + ((1 + col) * 4);
    for (int row = 0; row < 4; ++row) {
      sdf_shapes_data[base_idx_col + row] = obmat[col][row]; 
    }
  }

  active_shapes_count++;
}

static void draw_sdf_viewport_scene(gpu::FrameBuffer *active_fb, 
                                    SdfViewportStorage *storage,
                                    const float view_mat[4][4],
                                    const float proj_mat[4][4],
                                    bool is_persp) {
  if (!active_fb || !storage) {
    return;
  }

  /* 1. Allocate or Update Shape Texture */
  if (!shape_texture) {
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

  /* 2. Compute View-Projection and Inverses */
  float view_proj[4][4];
  mul_m4_m4m4(view_proj, proj_mat, view_mat);

  float inv_view_proj[4][4];
  invert_m4_m4(inv_view_proj, view_proj);

  float inv_view[4][4];
  invert_m4_m4(inv_view, view_mat);

  /* Camera World Position & Forward Direction */
  float cam_pos[3] = {inv_view[3][0], inv_view[3][1], inv_view[3][2]};
  float cam_forward[3] = {-inv_view[2][0], -inv_view[2][1], -inv_view[2][2]};
  normalize_v3(cam_forward);

  int vp[4] = {0, 0, 1, 1};
  GPU_framebuffer_viewport_get(active_fb, vp);
  float viewport_size[2] = {
      static_cast<float>(vp[2] > 0 ? vp[2] : 1),
      static_cast<float>(vp[3] > 0 ? vp[3] : 1)
  };

  /* Exact debug gradient palette matching the reference script */
  float max_dist = 200.0f;
  float debug_color_start[3] = {1.0f, 0.0f, 0.0f}; /* Red: Far from shapes */
  float debug_color_end[3]   = {0.0f, 0.0f, 1.0f}; /* Blue: Near shapes */

  /* 3. Push and isolate matrix stacks for 2D Screen Quad */
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

  GPU_depth_test(GPU_DEPTH_ALWAYS);
  GPU_depth_mask(true);

  immBindShader(sdf_shader);
  GPU_texture_bind(shape_texture, 0);

  /* Send Uniforms */
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
  GPUVertFormat *format = immVertexFormat();
  uint pos_attr = GPU_vertformat_attr_add(format, "position", gpu::VertAttrType::SFLOAT_32_32);

  immBegin(GPU_PRIM_TRIS, 6);
  immVertex2f(pos_attr, -1.0f, -1.0f);
  immVertex2f(pos_attr,  1.0f, -1.0f);
  immVertex2f(pos_attr, -1.0f,  1.0f);
  immVertex2f(pos_attr, -1.0f,  1.0f);
  immVertex2f(pos_attr,  1.0f, -1.0f);
  immVertex2f(pos_attr,  1.0f,  1.0f);
  immEnd();

  immUnbindProgram();
  GPU_texture_unbind(shape_texture);

  /* 4. Restore Matrix & Depth State */
  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_matrix_pop_projection();
  GPU_matrix_pop();
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
  storage->begin_sync();

  DEGObjectIterSettings deg_iter_settings{};
  deg_iter_settings.depsgraph = depsgraph;
  deg_iter_settings.flags = DEG_ITER_OBJECT_FLAG_LINKED_DIRECTLY | DEG_ITER_OBJECT_FLAG_VISIBLE;

  DEG_OBJECT_ITER_BEGIN (&deg_iter_settings, ob) {
    storage->sync_object_manual(ob);
  } DEG_OBJECT_ITER_END;

  //* Extract 3D Viewport Matrices */
  RegionView3D *rv3d = CTX_wm_region_view3d(context);
  ARegion *region = CTX_wm_region(context);

  float view_mat[4][4], proj_mat[4][4];
  bool is_persp = true;

  if (rv3d && region) {
    copy_m4_m4(view_mat, rv3d->viewmat);
    copy_m4_m4(proj_mat, rv3d->winmat);
    is_persp = (rv3d->winmat[3][3] == 0.0f);
  } else {
    GPU_matrix_model_view_get(view_mat);
    GPU_matrix_projection_get(proj_mat);
    is_persp = (proj_mat[3][3] == 0.0f);
  }

  draw_sdf_viewport_scene(active_fb, storage, view_mat, proj_mat, is_persp);
  delete storage;
}

}  // namespace blender::sdf_viewport

namespace blender {
RenderEngineType DRW_engine_viewport_sdf_viewport_type = {
    /* next */ nullptr,
    /* prev */ nullptr,
    /* idname */ "SDF_VIEWPORT",
    /* name */ N_("SDF Viewport"),
    /* flag */ blender::RE_USE_PREVIEW | blender::RE_USE_STEREO_VIEWPORT | blender::RE_USE_GPU_CONTEXT,
    /* update */ nullptr,
    /* render */ nullptr,
    /* render_frame_finish */ nullptr,
    /* draw */ nullptr,
    /* bake */ nullptr,
    /* view_update */ blender::sdf_viewport::sdf_viewport_view_update,
    /* view_draw */ blender::sdf_viewport::sdf_viewport_view_draw,
    /* update_script_node */ nullptr,
    /* update_render_passes */ nullptr,
    /* update_custom_camera */ nullptr,
    /* draw_engine */ (blender::DrawEngineType *)&blender::sdf_viewport::draw_engine_instance,
    /* rna_ext */ { nullptr, nullptr, nullptr },
};
}  // namespace blender