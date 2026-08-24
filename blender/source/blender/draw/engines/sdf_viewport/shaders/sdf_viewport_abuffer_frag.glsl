uniform sampler2D shapeDataTexture;

float4x4 get_matrix_from_texture(int shape_idx, int base_texel_x) {
    float4 c0 = texelFetch(shapeDataTexture, int2(base_texel_x + 0, shape_idx), 0);
    float4 c1 = texelFetch(shapeDataTexture, int2(base_texel_x + 1, shape_idx), 0);
    float4 c2 = texelFetch(shapeDataTexture, int2(base_texel_x + 2, shape_idx), 0);
    float4 c3 = texelFetch(shapeDataTexture, int2(base_texel_x + 3, shape_idx), 0);
    return float4x4(c0, c1, c2, c3);
}

float sdf_sphere(float3 p_world, float4x4 shape_world_matrix, float r) {
    float4x4 inv_shape_matrix = inverse(shape_world_matrix);
    float3 p_local = (inv_shape_matrix * float4(p_world, 1.0)).xyz;
    float3 scale_vec = float3(length(shape_world_matrix[0].xyz), 
                              length(shape_world_matrix[1].xyz), 
                              length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale; 
    return (length(p_local) - r) * min_scale;
}

float sdf_box(float3 p_world, float4x4 shape_world_matrix, float3 b_half_extents) {
    float4x4 inv_shape_matrix = inverse(shape_world_matrix);
    float3 p_local = (inv_shape_matrix * float4(p_world, 1.0)).xyz;
    float3 scale_vec = float3(length(shape_world_matrix[0].xyz), 
                              length(shape_world_matrix[1].xyz), 
                              length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    float3 q = abs(p_local) - b_half_extents;
    float d_local = length(max(q, float3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    return d_local * min_scale;
}

float sdf_torus(float3 p_world, float4x4 shape_world_matrix, float2 t_radii) {
    float4x4 inv_shape_matrix = inverse(shape_world_matrix);
    float3 p_local = (inv_shape_matrix * float4(p_world, 1.0)).xyz;
    float3 scale_vec = float3(length(shape_world_matrix[0].xyz), 
                              length(shape_world_matrix[1].xyz), 
                              length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    float2 q = float2(length(p_local.xz) - t_radii.x, p_local.y);
    float d_local = length(q) - t_radii.y;
    return d_local * min_scale;
}

float sdf_cylinder(float3 p_world, float4x4 shape_world_matrix, float2 rh) {
    float4x4 inv_shape_matrix = inverse(shape_world_matrix);
    float3 p_local = (inv_shape_matrix * float4(p_world, 1.0)).xyz;
    float3 scale_vec = float3(length(shape_world_matrix[0].xyz), 
                              length(shape_world_matrix[1].xyz), 
                              length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
    p_local /= min_scale;
    float2 d_abs = abs(float2(length(p_local.xz), p_local.y)) - rh;
    float d_local = min(max(d_abs.x, d_abs.y), 0.0) + length(max(d_abs, float2(0.0)));
    return d_local * min_scale;
}

float smooth_min(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float sdf_scene(float3 p_world) {
    float d_final = maxDist;
    float k_smooth = 0.3; 

    for (int i = 0; i < numActiveShapes; ++i) {
        float4 params_type = texelFetch(shapeDataTexture, int2(0, i), 0);
        float shape_type_id = params_type.x;
        float4x4 shape_world_mat = get_matrix_from_texture(i, 1);
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
        
        if (i == 0) {
            d_final = d_shape;
        } else {
            d_final = smooth_min(d_final, d_shape, k_smooth);
        }
    }
    return d_final;
}

float3 compute_normal(float3 p_world) {
    float eps = 0.001; 
    float2 h = float2(eps, 0.0);
    return normalize(float3(sdf_scene(p_world + h.xyy) - sdf_scene(p_world - h.xyy),
                            sdf_scene(p_world + h.yxy) - sdf_scene(p_world - h.yxy),
                            sdf_scene(p_world + h.yyx) - sdf_scene(p_world - h.yyx)));
}

float3 matcap_color(float3 world_normal) {
    float light = dot(world_normal, normalize(float3(0.5, 0.5, 1.0)));
    light = clamp(light, 0.0, 1.0);
    float3 base_color = float3(0.8, 0.85, 0.9);
    float3 highlight = float3(1.0, 1.0, 1.0);
    float3 rim = float3(0.3, 0.3, 0.35);
    float3 color = mix(rim, base_color, smoothstep(0.0, 0.5, light));
    color = mix(color, highlight, smoothstep(0.7, 1.0, light));
    return color;
}

void main() {
    float2 ndc = v_ndc;

    /* Unproject near and far NDC positions to world space */
    float4 p_near_clip = float4(ndc.x, ndc.y, -1.0, 1.0);
    float4 p_near_world = invViewProjectionMatrix * p_near_clip;
    p_near_world /= p_near_world.w;

    float4 p_far_clip = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 p_far_world = invViewProjectionMatrix * p_far_clip;
    p_far_world /= p_far_world.w;

    float3 ray_direction_world = normalize(p_far_world.xyz - p_near_world.xyz);
    float3 ray_origin_world;

    if (isPerspective != 0) {
        ray_origin_world = cameraPos_world;
    } else {
        /* In Orthographic view, project cameraPos_world onto the ray line so raymarching
         * starts near the camera view plane rather than far away at the ortho near clipping plane. */
        float3 ray_cam_plane = p_near_world.xyz + ray_direction_world * dot(cameraPos_world - p_near_world.xyz, ray_direction_world);
        ray_origin_world = ray_cam_plane - ray_direction_world * (maxDist * 0.25);
    }

    float t = 0.0; 
    float min_dist_to_surface = maxDist; 
    const int MAX_RAY_STEPS = 256; 
    const float HIT_EPSILON = 0.001; 

    for (int i = 0; i < MAX_RAY_STEPS; ++i) {
        float3 current_pos_world = ray_origin_world + t * ray_direction_world;
        float dist_sdf = sdf_scene(current_pos_world);
        min_dist_to_surface = min(min_dist_to_surface, abs(dist_sdf));

        if (abs(dist_sdf) < HIT_EPSILON) {
            float3 normal_world = compute_normal(current_pos_world);
            float3 color_shaded = matcap_color(normal_world); 
            float4 pos_clip = viewProjectionMatrix * float4(current_pos_world, 1.0);
            float depth_ndc = pos_clip.z / pos_clip.w; 
            gl_FragDepth = (depth_ndc * 0.5 + 0.5); 
            FragColor = float4(color_shaded, 1.0);
            return;
        }
        t += max(dist_sdf * 0.8, HIT_EPSILON * 0.5); 
        if (t > maxDist) {
            break;
        }
    }

    /* Distance gradient field background (Red when far, Blue when near) */
    float debug_norm = clamp(min_dist_to_surface / (maxDist * 0.1), 0.0, 1.0);
    debug_norm = pow(debug_norm, 0.5); 
    float3 color_bg = mix(debugColorEnd, debugColorStart, debug_norm); 
    FragColor = float4(color_bg, 1.0);
    gl_FragDepth = 1.0; 
}