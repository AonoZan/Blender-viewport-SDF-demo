set(INC
  ../../intern # Path to headers in the parent directory
  ../../../gpu
)

set(SRC
  sdf_viewport_engine.cc
  sdf_viewport_shader.cc
)

# This will add your source to the library used by the draw module
blender_add_lib(draw_engine_sdf_viewport "${SRC}" "${INC}" "" "")