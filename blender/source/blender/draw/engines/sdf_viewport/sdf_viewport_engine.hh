#pragma once

#include "DRW_render.hh"
#include "BLI_string_ref.hh"
#include <vector>

namespace blender {

/* Forward declare in blender namespace as it is no longer a global C struct */
struct RenderEngineType;

extern RenderEngineType DRW_engine_viewport_sdf_viewport_type;

namespace sdf_viewport {

struct SdfViewportStorage : public DrawEngine {
  gpu::Texture *dummy_render_target = nullptr;

  /* C++ dynamic storage for our SDF shape configurations */
  std::vector<float> sdf_shapes_data;
  int active_shapes_count = 0;

  SdfViewportStorage() = default;
  ~SdfViewportStorage() override = default;

  StringRefNull name_get() override { return "SDF Viewport"; }
  void init() override;
  void begin_sync() override;
  void object_sync(draw::ObjectRef &ob_ref, draw::Manager &manager) override;
  void sync_object_manual(Object *ob);
  void end_sync() override {}
  void draw(draw::Manager &manager) override;
};

struct Engine : public DrawEngine::Pointer {
  DrawEngine *create_instance() final;
  static void free_static();
};

}  // namespace sdf_viewport
}  // namespace blender