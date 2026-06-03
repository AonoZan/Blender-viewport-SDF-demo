import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
import numpy as np
import random # For randomizing shape types

# Debug gradient colors (RGB, range 0.0 to 1.0)
DEBUG_COLOR_START = (1.0, 0.0, 0.0)  # Red (far from shapes)
DEBUG_COLOR_END = (0.0, 0.0, 1.0)    # Blue (near shapes)

# Global variables
handle = None
shader_object = None
shape_texture = None
shape_texture_buffer = None

# --- Configuration for High Shape Count ---
TARGET_NUM_SHAPES = 10 # How many shapes to actually create and try to render
MAX_TEXTURE_SHAPES = TARGET_NUM_SHAPES # Max shapes our texture can hold (must be >= TARGET_NUM_SHAPES)

# Texture layout: 1 texel for type/params, 4 texels for matrix columns
SHAPE_TEXTURE_WIDTH = 1 + 4 # 5 texels wide per shape
SHAPE_TEXTURE_HEIGHT = MAX_TEXTURE_SHAPES # Each shape gets a row

# Create empties for SDF shapes in a grid
def create_many_sdf_empties(num_shapes_to_create):
    print(f"Attempting to create/update {num_shapes_to_create} SDF empties...")
    grid_side = int(np.ceil(np.sqrt(num_shapes_to_create)))
    spacing = 2.0  # Spacing between empties
    
    shape_types = ["sphere", "box", "torus", "cylinder"]
    count = 0

    for i in range(grid_side):
        if count >= num_shapes_to_create: break
        for j in range(grid_side):
            if count >= num_shapes_to_create: break
            
            name = f"SDF_Shape_{count:04d}" # Padded name for sorting
            
            chosen_type = random.choice(shape_types)
            params = []
            if chosen_type == "sphere": params = [random.uniform(0.3, 0.7)]
            elif chosen_type == "box": params = [random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]
            elif chosen_type == "torus": params = [random.uniform(0.4, 0.8), random.uniform(0.1, 0.3)]
            elif chosen_type == "cylinder": params = [random.uniform(0.2, 0.5), random.uniform(0.3, 0.7)]

            location_x = (j - grid_side / 2.0) * spacing
            location_y = (i - grid_side / 2.0) * spacing
            location_z = random.uniform(-1.0, 1.0) # Add Z variation
            location = (location_x, location_y, location_z)

            if name not in bpy.data.objects:
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
                empty = bpy.context.active_object
                empty.name = name
                empty.empty_display_size = 0.1 
            else:
                empty = bpy.data.objects[name]
                empty.location = location 

            empty["sdf_type"] = chosen_type
            empty["sdf_params"] = params
            
            empty.rotation_euler = (random.uniform(0, np.pi*2), random.uniform(0, np.pi*2), random.uniform(0, np.pi*2))
            empty.scale = (random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3))

            count += 1
    print(f"Finished creating/updating {count} SDF empties.")


# Generate vertices for a full-screen quad
def generate_quad_vertices():
    return [
        (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0),
        (-1.0, 1.0),  (1.0, -1.0), (1.0, 1.0),
    ]

# --- Modern Shader Source (Declarations are generated automatically by Info) ---
vertex_shader = '''
void main() {
    uv_frag = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
'''

fragment_shader = '''
mat4 get_matrix_from_texture(int shape_idx, int base_texel_x) {
    vec4 c0 = texelFetch(shapeDataTexture, ivec2(base_texel_x + 0, shape_idx), 0);
    vec4 c1 = texelFetch(shapeDataTexture, ivec2(base_texel_x + 1, shape_idx), 0);
    vec4 c2 = texelFetch(shapeDataTexture, ivec2(base_texel_x + 2, shape_idx), 0);
    vec4 c3 = texelFetch(shapeDataTexture, ivec2(base_texel_x + 3, shape_idx), 0);
    return mat4(c0, c1, c2, c3);
}

float sdf_sphere(vec3 p_world, mat4 shape_world_matrix, float r) {
    mat4 inv_shape_matrix = inverse(shape_world_matrix);
    vec3 p_local = (inv_shape_matrix * vec4(p_world, 1.0)).xyz;
    vec3 scale_vec = vec3(length(shape_world_matrix[0].xyz), 
                          length(shape_world_matrix[1].xyz), 
                          length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale; 
    return (length(p_local) - r) * min_scale;
}

float sdf_box(vec3 p_world, mat4 shape_world_matrix, vec3 b_half_extents) {
    mat4 inv_shape_matrix = inverse(shape_world_matrix);
    vec3 p_local = (inv_shape_matrix * vec4(p_world, 1.0)).xyz;
    vec3 scale_vec = vec3(length(shape_world_matrix[0].xyz), 
                          length(shape_world_matrix[1].xyz), 
                          length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    vec3 q = abs(p_local) - b_half_extents;
    float d_local = length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    return d_local * min_scale;
}

float sdf_torus(vec3 p_world, mat4 shape_world_matrix, vec2 t_radii) {
    mat4 inv_shape_matrix = inverse(shape_world_matrix);
    vec3 p_local = (inv_shape_matrix * vec4(p_world, 1.0)).xyz;
    vec3 scale_vec = vec3(length(shape_world_matrix[0].xyz), 
                          length(shape_world_matrix[1].xyz), 
                          length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    vec2 q = vec2(length(p_local.xz) - t_radii.x, p_local.y);
    float d_local = length(q) - t_radii.y;
    return d_local * min_scale;
}

float sdf_cylinder(vec3 p_world, mat4 shape_world_matrix, vec2 rh) {
    mat4 inv_shape_matrix = inverse(shape_world_matrix);
    vec3 p_local = (inv_shape_matrix * vec4(p_world, 1.0)).xyz;
    vec3 scale_vec = vec3(length(shape_world_matrix[0].xyz), 
                          length(shape_world_matrix[1].xyz), 
                          length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    vec2 d_abs = abs(vec2(length(p_local.xz), p_local.y)) - rh;
    float d_local = min(max(d_abs.x, d_abs.y), 0.0) + length(max(d_abs, vec2(0.0)));
    return d_local * min_scale;
}

float smooth_min(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float sdf_scene(vec3 p_world) {
    float d_final = maxDist;
    float k_smooth = 0.3; 

    for (int i = 0; i < numActiveShapes; ++i) {
        vec4 params_type = texelFetch(shapeDataTexture, ivec2(0, i), 0);
        float shape_type_id = params_type.x;
        mat4 shape_world_mat = get_matrix_from_texture(i, 1);
        float d_shape = maxDist;
        if (shape_type_id < 0.5) { 
            d_shape = sdf_sphere(p_world, shape_world_mat, params_type.y);
        } else if (shape_type_id < 1.5) {
            d_shape = sdf_box(p_world, shape_world_mat, params_type.yzw);
        } else if (shape_type_id < 2.5) {
            d_shape = sdf_torus(p_world, shape_world_mat, params_type.yz);
        } else if (shape_type_id < 3.5) {
            d_shape = sdf_cylinder(p_world, shape_world_mat, params_type.yz);
        }
        d_final = smooth_min(d_final, d_shape, k_smooth);
    }
    return d_final;
}

vec3 compute_normal(vec3 p_world) {
    float eps = 0.001; 
    vec2 h = vec2(eps, 0.0);
    return normalize( vec3( sdf_scene(p_world + h.xyy) - sdf_scene(p_world - h.xyy),
                           sdf_scene(p_world + h.yxy) - sdf_scene(p_world - h.yxy),
                           sdf_scene(p_world + h.yyx) - sdf_scene(p_world - h.yyx) ) );
}

vec3 matcap_color(vec3 world_normal) {
    float light = dot(world_normal, normalize(vec3(0.5, 0.5, 1.0)));
    light = clamp(light, 0.0, 1.0);
    vec3 base_color = vec3(0.8, 0.85, 0.9);
    vec3 highlight = vec3(1.0, 1.0, 1.0);
    vec3 rim = vec3(0.3, 0.3, 0.35);
    vec3 color = mix(rim, base_color, smoothstep(0.0, 0.5, light));
    color = mix(color, highlight, smoothstep(0.7, 1.0, light));
    return color;
}

void main() {
    vec2 screen_uv = gl_FragCoord.xy / viewportSize;
    vec2 ndc = screen_uv * 2.0 - 1.0;          

    vec3 ray_origin_world;
    vec3 ray_direction_world;

    if (isPerspective) {
        ray_origin_world = cameraPos_world;
        vec4 far_clip = vec4(ndc.x, ndc.y, 1.0, 1.0); 
        vec4 far_world = invViewProjectionMatrix * far_clip;
        ray_direction_world = normalize(far_world.xyz / far_world.w - ray_origin_world);
    } else { 
        vec4 near_clip = vec4(ndc.x, ndc.y, -1.0, 1.0);
        vec4 near_world = invViewProjectionMatrix * near_clip; 
        ray_origin_world = near_world.xyz / near_world.w;
        ray_direction_world = cameraForward_world;
    }

    float t = 0.0; 
    float min_dist_to_surface = maxDist; 
    const int MAX_RAY_STEPS = 256; 
    const float HIT_EPSILON = 0.001; 

    for (int i = 0; i < MAX_RAY_STEPS; ++i) {
        vec3 current_pos_world = ray_origin_world + t * ray_direction_world;
        float dist_sdf = sdf_scene(current_pos_world);
        min_dist_to_surface = min(min_dist_to_surface, abs(dist_sdf));

        if (abs(dist_sdf) < HIT_EPSILON) {
            vec3 normal_world = compute_normal(current_pos_world);
            vec3 color_shaded = matcap_color(normal_world); 
            vec4 pos_clip = viewProjectionMatrix * vec4(current_pos_world, 1.0);
            float depth_ndc = pos_clip.z / pos_clip.w; 
            gl_FragDepth = (depth_ndc * 0.5 + 0.5); 
            FragColor = vec4(color_shaded, 1.0);
            return;
        }
        t += max(dist_sdf * 0.8, HIT_EPSILON * 0.5); 
        if (t > maxDist) break; 
    }

    float debug_norm = clamp(min_dist_to_surface / (maxDist * 0.1), 0.0, 1.0);
    debug_norm = pow(debug_norm, 0.5); 
    vec3 color_bg = mix(debugColorEnd, debugColorStart, debug_norm); 
    FragColor = vec4(color_bg, 1.0);
    gl_FragDepth = 1.0; 
}
'''

# --- GPU Resource Management ---
def update_shape_texture_resources(objects_list):
    global shape_texture, shape_texture_buffer
    num_valid_shapes = 0
    
    data_np = np.zeros(MAX_TEXTURE_SHAPES * SHAPE_TEXTURE_WIDTH * 4, dtype=np.float32)
    type_map = {"sphere": 0.0, "box": 1.0, "torus": 2.0, "cylinder": 3.0}

    for i, obj in enumerate(objects_list):
        if i >= MAX_TEXTURE_SHAPES: break 

        sdf_type_str = obj.get("sdf_type", "sphere")
        sdf_params_list = obj.get("sdf_params", [1.0])
        type_id = type_map.get(sdf_type_str, 0.0)
        
        current_params = [type_id]
        current_params.extend(sdf_params_list)
        padded_params_for_texel = (current_params + [0.0, 0.0, 0.0])[:4]
        
        base_idx_params = (i * SHAPE_TEXTURE_WIDTH * 4) + (0 * 4)
        data_np[base_idx_params : base_idx_params + 4] = padded_params_for_texel
        
        matrix = obj.matrix_world.transposed()
        for col_idx in range(4):
            base_idx_matrix_col = (i * SHAPE_TEXTURE_WIDTH * 4) + ((1 + col_idx) * 4)
            data_np[base_idx_matrix_col : base_idx_matrix_col + 4] = matrix[col_idx]
        num_valid_shapes += 1

    data_list = data_np.tolist()
    buffer_len_elements = len(data_list)

    try:
        # Recreate the GPU buffer and GPU texture every frame to force data update on the GPU
        shape_texture_buffer = gpu.types.Buffer('FLOAT', buffer_len_elements)
        shape_texture_buffer[:] = data_list
    except Exception as e:
        print(f"FATAL: Error during gpu.types.Buffer creation or population: {e}")
        shape_texture_buffer = None; return None, 0

    tex_dims = (SHAPE_TEXTURE_WIDTH, SHAPE_TEXTURE_HEIGHT) 
    try:
        shape_texture = gpu.types.GPUTexture(size=tex_dims, format='RGBA32F', data=shape_texture_buffer)
        if hasattr(shape_texture, 'interpolation'):
            shape_texture.interpolation = 'Closest'
        elif hasattr(shape_texture, 'use_interpolation'):
            shape_texture.use_interpolation = False
    except Exception as e:
        print(f"FATAL: Error creating/recreating gpu.types.GPUTexture: {e}")
        shape_texture = None; return None, 0
            
    return shape_texture, num_valid_shapes


# --- Safe Uniform Binding Helpers ---
def safe_uniform_float(shader, name, value):
    try: shader.uniform_float(name, value)
    except Exception: pass # Ignore if optimized away by driver

def safe_uniform_int(shader, name, value):
    try: shader.uniform_int(name, value)
    except Exception: pass

def safe_uniform_bool(shader, name, value):
    try: shader.uniform_bool(name, value)
    except Exception: pass

def safe_uniform_sampler(shader, name, value):
    try: shader.uniform_sampler(name, value)
    except Exception: pass


# --- Draw Callback ---
def draw():
    global shader_object

    if shader_object is None: return

    context = bpy.context; region = context.region; region_3d = context.space_data.region_3d
    if not all([context, region, region_3d]) or region.width == 0 or region.height == 0: return

    sdf_objects = []
    for obj in bpy.data.objects:
        if obj.name.startswith("SDF_Shape_") and obj.get("sdf_type") and obj.get("sdf_params"):
            sdf_objects.append(obj)
    sdf_objects.sort(key=lambda o: o.name)
            
    texture, num_active_shapes = update_shape_texture_resources(sdf_objects[:MAX_TEXTURE_SHAPES])

    if texture is None or num_active_shapes == 0: return

    is_perspective = region_3d.is_perspective
    view_matrix = region_3d.view_matrix
    proj_matrix = region_3d.window_matrix
    view_projection_matrix = proj_matrix @ view_matrix
    try:
        inv_view_projection_matrix = view_projection_matrix.inverted()
        inv_view_matrix = view_matrix.inverted()
        camera_pos_world = inv_view_matrix.translation
        
        # Extract and negate the camera's world-space Z-axis
        camera_forward = -Vector((inv_view_matrix[0][2], inv_view_matrix[1][2], inv_view_matrix[2][2]))
        camera_forward.normalize()
    except ValueError: print("Warning: View matrix not invertible."); return
        
    viewport_size = (float(region.width), float(region.height))
    max_dist_val = 200.0

    vertices = generate_quad_vertices()
    batch = batch_for_shader(shader_object, 'TRIS', {"position": vertices})

    shader_object.bind()
    
    # Use safe binding methods and pass scalar values for non-array uniforms
    safe_uniform_float(shader_object, "cameraPos_world", camera_pos_world)
    safe_uniform_float(shader_object, "cameraForward_world", camera_forward)
    safe_uniform_float(shader_object, "maxDist", max_dist_val)
    safe_uniform_bool(shader_object, "isPerspective", is_perspective)
    safe_uniform_float(shader_object, "viewProjectionMatrix", view_projection_matrix)
    safe_uniform_float(shader_object, "invViewProjectionMatrix", inv_view_projection_matrix)
    safe_uniform_float(shader_object, "viewportSize", viewport_size)
    safe_uniform_float(shader_object, "debugColorStart", DEBUG_COLOR_START)
    safe_uniform_float(shader_object, "debugColorEnd", DEBUG_COLOR_END)
    safe_uniform_sampler(shader_object, "shapeDataTexture", texture)
    safe_uniform_int(shader_object, "numActiveShapes", num_active_shapes)

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')
    batch.draw(shader_object)
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)


# --- Handler Management ---
def scene_update_handler(scene):
    if bpy.context.window_manager:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for region_iter in area.regions:
                        if region_iter.type == 'WINDOW': region_iter.tag_redraw()

def register():
    global handle, shader_object, shape_texture, shape_texture_buffer
    unregister()
    try:
        # Create metadata struct for shader compiler
        info = gpu.types.GPUShaderCreateInfo()
        
        # 1. Define inputs
        info.vertex_in(0, 'VEC2', "position")
        
        # 2. Define stage interface (Vertex Out -> Fragment In)
        interface = gpu.types.GPUStageInterfaceInfo("my_interface")
        interface.smooth('VEC2', "uv_frag")
        info.vertex_out(interface)
        
        # 3. Define outputs
        info.fragment_out(0, 'VEC4', "FragColor")
        
        # 4. Define uniform layout (using push constants)
        info.push_constant('VEC3', "cameraPos_world")
        info.push_constant('VEC3', "cameraForward_world")
        info.push_constant('FLOAT', "maxDist")
        info.push_constant('BOOL', "isPerspective")
        info.push_constant('MAT4', "viewProjectionMatrix")
        info.push_constant('MAT4', "invViewProjectionMatrix")
        info.push_constant('VEC2', "viewportSize")
        info.push_constant('VEC3', "debugColorStart")
        info.push_constant('VEC3', "debugColorEnd")
        info.push_constant('INT', "numActiveShapes")
        
        # 5. Define samplers (FLOAT_2D is required for float-based textures)
        info.sampler(0, 'FLOAT_2D', "shapeDataTexture")
        
        # 6. Assign sources
        info.vertex_source(vertex_shader)
        info.fragment_source(fragment_shader)
        
        # 7. Compile
        shader_object = gpu.shader.create_from_info(info)
        print("Shader compiled successfully.")
    except Exception as e:
        print(f"FATAL: Shader compilation failed: {e}")
        shader_object = None; return
        
    shape_texture = None
    shape_texture_buffer = None
    handle = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
    print("Draw handler registered.")
    if scene_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(scene_update_handler)
        print("Depsgraph update handler registered.")

def unregister():
    global handle, shader_object, shape_texture, shape_texture_buffer
    if handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        handle = None
    if 'scene_update_handler' in globals() and scene_update_handler in bpy.app.handlers.depsgraph_update_post:
        try: bpy.app.handlers.depsgraph_update_post.remove(scene_update_handler)
        except ValueError: pass
    shader_object = None
    shape_texture = None
    shape_texture_buffer = None
    print("Handlers unregistered and GPU resource references cleared.")


# --- Script Execution ---
if __name__ == "__main__":
    unregister()
    create_many_sdf_empties(TARGET_NUM_SHAPES)
    register()