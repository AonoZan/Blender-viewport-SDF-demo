## SDF demo done with Blender GPU module

Copy python code and paste it in text editor, run the code to see render. You can duplicate, move, rotate and scale shapes. Performance is not that great with the more shapes you add. Ive tested with 100 shapes on RTX 3070 and its sluggish, 1000 shapes make FPS <1 so be aware of this.

![Project Banner](./Screenshot.png)



## Native Blender Demo added (Tested only on Blender 5.0)

To see this demo clone Blender source code from git compile and make sure you can run it.
Then copy folder "sdf_viewport" to the /blender/source/blender/draw/engines/.

/blender/source/blender/draw/intern/draw_context.cc
Include headers at the top:
```
#include "engines/overlay/overlay_engine.h"
#include "engines/select/select_engine.hh"
#include "engines/workbench/workbench_engine.h"
#include "engines/sdf_viewport/sdf_viewport_engine.hh" <-- Include this
#include "engines/sdf_viewport/sdf_viewport_shader.hh" <--

#include "GPU_context.hh"
```
blender/source/blender/draw/CMakeLists.txt
```
  engines/overlay/overlay_mode_transfer.cc
  engines/overlay/overlay_shader.cc
  engines/overlay/overlay_shape.cc
  engines/sdf_viewport/sdf_viewport_engine.cc <---
  engines/sdf_viewport/sdf_viewport_shader.cc <---
```
```
  engines/image/shaders/image_engine_image_tiled_frag.glsl
  engines/image/shaders/image_engine_lib.glsl

  engines/image/image_shader_shared.hh

  # This test doesn't depend on gpu_shader_test, so it's always included.
  intern/shaders/draw_curves_test.glsl

  engines/sdf_viewport/shaders/sdf_viewport_abuffer_frag.glsl <---
  engines/sdf_viewport/shaders/sdf_viewport_sdf_vert.glsl <---
)
```
```
  engines/image
  engines/image/shaders
  engines/image/shaders/infos
  engines/sdf_viewport
  engines/sdf_viewport/shaders
```
/blender/scripts/startup/bl_ui/properties_render.py
Add it to COMPAT_ENGINE list in:
RENDER_PT_color_management
RENDER_PT_color_management_working_space
RENDER_PT_color_management_advanced
RENDER_PT_color_management_curves
RENDER_PT_color_management_white_balance
```
COMPAT_ENGINES = {
        'BLENDER_RENDER',
        'BLENDER_EEVEE',
        'BLENDER_WORKBENCH',
        'SDF_VIEWPORT',
    }
```
Add new classes and register them:

```
class OBJECT_OT_sdf_viewport_spawn_sdf(Operator):
    """Spawn and randomize coordinates of SDF shapes in the scene"""
    bl_idname = "object.sdf_viewport_spawn_sdf"
    bl_label = "Generate SDF Objects"
    bl_options = {'REGISTER', 'UNDO'}

    num_shapes: bpy.props.IntProperty(
        name="Target Shapes",
        default=10,
        min=1,
        max=64,
        description="Number of primitives to generate"
    )

    def execute(self, context):
        num_shapes_to_create = self.num_shapes
        grid_side = math.ceil(math.sqrt(num_shapes_to_create))
        spacing = 2.0
        
        shape_types = ["sphere", "box", "torus", "cylinder"]
        count = 0

        for i in range(grid_side):
            if count >= num_shapes_to_create:
                break
            for j in range(grid_side):
                if count >= num_shapes_to_create:
                    break
                
                name = f"SDF_Shape_{count:04d}"
                chosen_type = random.choice(shape_types)
                
                # Parameters matching GLSL expectation layouts
                params = []
                if chosen_type == "sphere":
                    params = [random.uniform(0.3, 0.7)]
                elif chosen_type == "box":
                    params = [random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]
                elif chosen_type == "torus":
                    params = [random.uniform(0.4, 0.8), random.uniform(0.1, 0.3)]
                elif chosen_type == "cylinder":
                    params = [random.uniform(0.2, 0.5), random.uniform(0.3, 0.7)]

                location_x = (j - grid_side / 2.0) * spacing
                location_y = (i - grid_side / 2.0) * spacing
                location_z = random.uniform(-1.0, 1.0)
                location = (location_x, location_y, location_z)

                # Create or locate existing Empty object
                if name not in bpy.data.objects:
                    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
                    empty = context.active_object
                    empty.name = name
                    empty.empty_display_size = 0.1 
                else:
                    empty = bpy.data.objects[name]
                    empty.location = location 

                # Populate dynamic attributes (IDProperties)
                empty["sdf_type"] = chosen_type
                empty["sdf_params"] = params
                
                # Apply random rotation and scale values
                empty.rotation_euler = (random.uniform(0, math.pi*2), random.uniform(0, math.pi*2), random.uniform(0, math.pi*2))
                empty.scale = (random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3))

                count += 1
                
        self.report({'INFO'}, f"SDF Viewport: Synced {count} primitives.")
        return {'FINISHED'}


class RENDER_PT_sdf_viewport_settings(Panel):
    """Panel options in the Render Tab when SDF Viewport is selected"""
    bl_label = "SDF Viewport Settings"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'SDF_VIEWPORT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.label(text="SDF Generator", icon='MESH_CUBE')
        
        # Display Spawning Controls
        op = col.operator("object.sdf_viewport_spawn_sdf", text="Spawn / Update Shapes", icon='ADD')
        op.num_shapes = 12
```
```
classes = (
....
    RENDER_PT_color_management_white_balance,
    RENDER_PT_color_management_working_space,
    RENDER_PT_color_management_advanced,
    OBJECT_OT_sdf_viewport_spawn_sdf,
    RENDER_PT_sdf_viewport_settings
)
```
/blender/source/blender/draw/engines/overlay/overlay_instance.cc
Enable depth test for this new render engine so that grid can and other Viewport elements can use it.

```
    const bool viewport_uses_eevee = STREQ(
        ED_view3d_engine_type(state.scene, state.v3d->shading.type)->idname,
        RE_engine_id_BLENDER_EEVEE);
    const bool viewport_uses_sdf = STREQ( <-- add bool
        ED_view3d_engine_type(state.scene, state.v3d->shading.type)->idname,
        "SDF_VIEWPORT");

    state.is_render_depth_available = viewport_uses_workbench ||
                                      (viewport_uses_eevee && !use_resolution_scaling) ||
                                      viewport_uses_sdf; <-- and add it to depth test

```

Recompile.

When you run compiled Blender, you can select "SDF Viewport" from render engine dropdown.

![SDF Native Render](./Screenshot_sdf_native.png)
