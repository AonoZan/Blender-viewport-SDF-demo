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
    
    float d_local = length(p_local) - r;
    return d_local * min_scale;
}

float sdf_box(float3 p_world, float4x4 shape_world_matrix, float3 b_half_extents) {
    float4x4 inv_shape_matrix = inverse(shape_world_matrix);
    float3 p_local = (inv_shape_matrix * float4(p_world, 1.0)).xyz;
    float3 scale_vec = float3(length(shape_world_matrix[0].xyz), 
                              length(shape_world_matrix[1].xyz), 
                              length(shape_world_matrix[2].xyz));
    float min_scale = max(0.0001, min(scale_vec.x, min(scale_vec.y, scale_vec.z)));
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
    float2 d_abs = abs(float2(length(p_local.xz), p_local.y)) - rh;
    float d_local = min(max(d_abs.x, d_abs.y), 0.0) + length(max(d_abs, float2(0.0)));
    return d_local * min_scale;
}

float smooth_min(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float get_shape_selection_status(int shape_idx) {
    if (shape_idx < 0 || shape_idx >= numActiveShapes) {
        return 0.0;
    }
    float4 sel_data = texelFetch(shapeDataTexture, int2(5, shape_idx), 0);
    return sel_data.x;
}

float eval_shape_dist(int i, float3 p_world) {
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
    return d_shape;
}

// Evaluates overall scene distance while smoothly grouping shapes by selection mode
float eval_scene_grouped(float3 p_world, out float out_d_active, out float out_d_selected, out float out_d_unselected) {
    float d_scene = maxDist;
    out_d_active = maxDist;
    out_d_selected = maxDist;
    out_d_unselected = maxDist;
    
    float k_smooth = 0.3;
    bool has_act = false;
    bool has_sel = false;
    bool has_unsel = false;

    for (int i = 0; i < numActiveShapes; ++i) {
        float d_shape = eval_shape_dist(i, p_world);
        float sel_status = get_shape_selection_status(i);

        if (sel_status > 1.5) {
            out_d_active = has_act ? smooth_min(out_d_active, d_shape, k_smooth) : d_shape;
            has_act = true;
        } else if (sel_status > 0.0) {
            out_d_selected = has_sel ? smooth_min(out_d_selected, d_shape, k_smooth) : d_shape;
            has_sel = true;
        } else {
            out_d_unselected = has_unsel ? smooth_min(out_d_unselected, d_shape, k_smooth) : d_shape;
            has_unsel = true;
        }

        if (i == 0) {
            d_scene = d_shape;
        } else {
            d_scene = smooth_min(d_scene, d_shape, k_smooth);
        }
    }

    return d_scene;
}

float sdf_scene(float3 p_world) {
    if (numActiveShapes == 0) {
        return maxDist;
    }

    float d_final = maxDist;
    float k_smooth = 0.3;
    for (int i = 0; i < numActiveShapes; ++i) {
        float d_shape = eval_shape_dist(i, p_world);
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

float compute_pixel_size(float3 p_world, float t) {
    float view_up_len = length(invViewProjectionMatrix[1].xyz);
    float vp_h = max(viewportSize.y, 1.0);
    if (isPerspective != 0) {
        float dist_from_cam = max(dot(p_world - cameraPos_world, cameraForward_world), 0.001);
        return (2.0 * dist_from_cam * view_up_len) / vp_h;
    } else {
        return (2.0 * view_up_len) / vp_h;
    }
}

void main() {
    if (numActiveShapes == 0) {
        discard;
    }

    float2 ndc = v_ndc;
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
        float3 ray_cam_plane = p_near_world.xyz + ray_direction_world * dot(cameraPos_world - p_near_world.xyz, ray_direction_world);
        ray_origin_world = ray_cam_plane - ray_direction_world * (maxDist * 0.25);
    }

    float t = 0.0;
    const int MAX_RAY_STEPS = 256;
    float outline_width_px = 1.75;

    float global_act_min_d = 100000.0;
    float global_sel_min_d = 100000.0;
    float act_hit_t = 0.0;
    float sel_hit_t = 0.0;

    for (int i = 0; i < MAX_RAY_STEPS; ++i) {
        float3 current_pos_world = ray_origin_world + t * ray_direction_world;
        
        float pix_sz = compute_pixel_size(current_pos_world, t);
        float px = max(pix_sz, 0.000001);
        float current_hit_epsilon = max(pix_sz * 0.25, 0.0001);

        float d_act, d_sel, d_unsel;
        float dist_sdf = eval_scene_grouped(current_pos_world, d_act, d_sel, d_unsel);

        // Track minimum distance to group silhouettes along the ray
        if (d_act < maxDist) {
            float act_d_px = abs(d_act) / px;
            if (act_d_px < global_act_min_d) {
                global_act_min_d = act_d_px;
                act_hit_t = t;
            }
        }
        if (d_sel < maxDist) {
            float sel_d_px = abs(d_sel) / px;
            if (sel_d_px < global_sel_min_d) {
                global_sel_min_d = sel_d_px;
                sel_hit_t = t;
            }
        }

        // --- Surface Hit ---
        if (dist_sdf < current_hit_epsilon) {
            float3 normal_world = compute_normal(current_pos_world);
            float3 color_shaded = matcap_color(normal_world);
            float4 pos_clip = viewProjectionMatrix * float4(current_pos_world, 1.0);
            float depth_ndc = pos_clip.z / pos_clip.w;
            gl_FragDepth = clamp(depth_ndc * 0.5 + 0.5, 0.0, 1.0);

            float3 final_color = color_shaded;

            // Determine if hit surface belongs to active or selected group
            bool is_active_surface = (d_act <= d_sel + 0.001) && (d_act <= d_unsel + 0.001);
            bool is_selected_surface = (d_sel <= d_act + 0.001) && (d_sel <= d_unsel + 0.001);

            // Compute silhouette distance across blended group boundaries
            float active_outline_d = is_active_surface ? (abs(d_act - min(d_sel, d_unsel)) / px) : global_act_min_d;
            float selected_outline_d = is_selected_surface ? (abs(d_sel - min(d_act, d_unsel)) / px) : global_sel_min_d;

            if (active_outline_d < outline_width_px) {
                float3 active_col = float3(1.0, 0.666, 0.117);
                float alpha = smoothstep(outline_width_px, max(outline_width_px - 1.0, 0.0), active_outline_d);
                final_color = mix(final_color, active_col, alpha);
            } else if (selected_outline_d < outline_width_px) {
                float3 sel_col = float3(0.941, 0.353, 0.047);
                float alpha = smoothstep(outline_width_px, max(outline_width_px - 1.0, 0.0), selected_outline_d);
                final_color = mix(final_color, sel_col, alpha);
            }

            FragColor = float4(final_color, 1.0);
            return;
        }

        float step_factor = (global_act_min_d < outline_width_px * 3.0 || global_sel_min_d < outline_width_px * 3.0) ? 0.35 : 1.0;
        t += max(dist_sdf * step_factor, current_hit_epsilon);

        if (t > maxDist) {
            break;
        }
    }

    // --- Silhouette Pass (Rays missing surfaces) ---
    if (global_act_min_d < outline_width_px) {
        float3 sel_pos = ray_origin_world + act_hit_t * ray_direction_world;
        float4 pos_clip = viewProjectionMatrix * float4(sel_pos, 1.0);
        gl_FragDepth = clamp((pos_clip.z / pos_clip.w) * 0.5 + 0.5, 0.0, 1.0);

        float3 active_col = float3(1.0, 0.666, 0.117);
        float alpha = smoothstep(outline_width_px, max(outline_width_px - 1.0, 0.0), global_act_min_d);
        FragColor = float4(active_col, alpha);
        return;
    } else if (global_sel_min_d < outline_width_px) {
        float3 sel_pos = ray_origin_world + sel_hit_t * ray_direction_world;
        float4 pos_clip = viewProjectionMatrix * float4(sel_pos, 1.0);
        gl_FragDepth = clamp((pos_clip.z / pos_clip.w) * 0.5 + 0.5, 0.0, 1.0);

        float3 sel_col = float3(0.941, 0.353, 0.047);
        float alpha = smoothstep(outline_width_px, max(outline_width_px - 1.0, 0.0), global_sel_min_d);
        FragColor = float4(sel_col, alpha);
        return;
    }

    discard;
}