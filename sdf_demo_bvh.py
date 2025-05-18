import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix, Euler
import numpy as np
import random
from math import inf

# --- Globals and Config ---
handle = None
shader_object = None
shape_texture = None
shape_texture_buffer = None
bvh_aabb_texture = None
bvh_aabb_buffer = None
bvh_meta_texture = None
bvh_meta_buffer = None
bvh_leaf_prims_texture = None
bvh_leaf_prims_buffer = None

TARGET_NUM_SHAPES = 100
MAX_TEXTURE_SHAPES = 500
SHAPE_TEXTURE_WIDTH = 5
SHAPE_TEXTURE_HEIGHT = MAX_TEXTURE_SHAPES
bvh_flat_nodes = []
node_counter = 0
MAX_LEAF_PRIMITIVES = 4

DEFAULT_RIM_CENTER_BALANCE = 0.5
DEFAULT_RIM_SHARPNESS = 3.0

# --- Helper: AABB Class ---
class AABB:
    def __init__(self, min_p=None, max_p=None):
        if min_p is None:
            self.min = Vector((float('inf'), float('inf'), float('inf')))
        else:
            self.min = Vector(min_p)
        if max_p is None:
            self.max = Vector((-float('inf'), -float('inf'), -float('inf')))
        else:
            self.max = Vector(max_p)

    def extend(self, point_or_aabb):
        if isinstance(point_or_aabb, AABB):
            self.min.x = min(self.min.x, point_or_aabb.min.x)
            self.min.y = min(self.min.y, point_or_aabb.min.y)
            self.min.z = min(self.min.z, point_or_aabb.min.z)
            self.max.x = max(self.max.x, point_or_aabb.max.x)
            self.max.y = max(self.max.y, point_or_aabb.max.y)
            self.max.z = max(self.max.z, point_or_aabb.max.z)
        else:
            p = Vector(point_or_aabb)
            self.min.x = min(self.min.x, p.x)
            self.min.y = min(self.min.y, p.y)
            self.min.z = min(self.min.z, p.z)
            self.max.x = max(self.max.x, p.x)
            self.max.y = max(self.max.y, p.y)
            self.max.z = max(self.max.z, p.z)

    def surface_area(self):
        d = self.max - self.min
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z) if d.x >= 0 and d.y >= 0 and d.z >= 0 else 0.0

    def centroid(self):
        return (self.min + self.max) * 0.5

    def __repr__(self):
        return f"AABB(min={tuple(self.min)}, max={tuple(self.max)})"

# --- AABB Calculation ---
def get_sphere_aabb(m: Matrix, r: float) -> AABB:
    c = m.translation
    sx = m.col[0].xyz.length
    sy = m.col[1].xyz.length
    sz = m.col[2].xyz.length
    max_scale = max(sx, sy, sz, 1e-5)
    scaled_radius = r * max_scale
    return AABB(c - Vector((scaled_radius, scaled_radius, scaled_radius)),
                c + Vector((scaled_radius, scaled_radius, scaled_radius)))

def get_box_aabb(m: Matrix, he: Vector) -> AABB:
    corners = [
        Vector((s_x * he.x, s_y * he.y, s_z * he.z))
        for s_x in [-1, 1] for s_y in [-1, 1] for s_z in [-1, 1]
    ]
    aabb = AABB()
    for corner in corners:
        transformed = m @ corner
        aabb.extend(transformed)
    return aabb

def get_cylinder_aabb(m: Matrix, r: float, hh: float) -> AABB:
    points = [
        Vector((s_x * r, s_y * hh, s_z * r))
        for s_x in [-1, 1] for s_y in [-1, 1] for s_z in [-1, 1]
    ]
    aabb = AABB()
    for p in points:
        transformed = m @ p
        aabb.extend(transformed)
    return aabb

def get_torus_aabb(m: Matrix, maj_r: float, min_r: float) -> AABB:
    outer_radius = maj_r + min_r
    points = [
        Vector((s_x * outer_radius, 0.0, s_z * outer_radius))
        for s_x in [-1, 1] for s_z in [-1, 1]
    ] + [
        Vector((s_x * maj_r, s_y * min_r, s_z * maj_r))
        for s_x in [-1, 1] for s_y in [-1, 1] for s_z in [-1, 1]
    ]
    aabb = AABB()
    for p in points:
        transformed = m @ p
        aabb.extend(transformed)
    return aabb

class SDFObjectData:
    def __init__(self, bo, obj_idx):
        self.bl_object = bo
        self.obj_idx = obj_idx
        self.name = bo.name
        self.sdf_type_str = bo.get("sdf_type", "sphere")
        self.sdf_params = list(bo.get("sdf_params", [1.0]))
        self.matrix_world = bo.matrix_world.copy()
        self.aabb = self.calculate_aabb()
        self.centroid = self.aabb.centroid()

    def calculate_aabb(self) -> AABB:
        if self.sdf_type_str == "sphere":
            return get_sphere_aabb(self.matrix_world, self.sdf_params[0])
        elif self.sdf_type_str == "box":
            return get_box_aabb(self.matrix_world, Vector(self.sdf_params))
        elif self.sdf_type_str == "cylinder":
            return get_cylinder_aabb(self.matrix_world, self.sdf_params[0], self.sdf_params[1])
        elif self.sdf_type_str == "torus":
            return get_torus_aabb(self.matrix_world, self.sdf_params[0], self.sdf_params[1])
        return AABB(
            self.matrix_world.translation - Vector((0.1, 0.1, 0.1)),
            self.matrix_world.translation + Vector((0.1, 0.1, 0.1))
        )

    def __repr__(self):
        return f"SDFObj(n='{self.name}', t='{self.sdf_type_str}', aabb={self.aabb})"

class BVHNode:
    def __init__(self):
        self.aabb = AABB()
        self.is_leaf = False
        self.primitive_indices = []
        self.left_child_idx = -1
        self.right_child_idx = -1
        self.node_idx = -1
        self.leaf_prims_offset = -1

def compute_sah_split(objs, axis, split_pos):
    left_aabb = AABB()
    right_aabb = AABB()
    left_count = 0
    right_count = 0

    for obj in objs:
        centroid = obj.centroid[axis]
        if centroid < split_pos:
            left_aabb.extend(obj.aabb)
            left_count += 1
        else:
            right_aabb.extend(obj.aabb)
            right_count += 1

    if left_count == 0 or right_count == 0:
        return float('inf'), None, None, 0, 0

    left_area = left_aabb.surface_area()
    right_area = right_aabb.surface_area()
    cost = 1.0 + left_area * left_count + right_area * right_count
    return cost, left_aabb, right_aabb, left_count, right_count

def build_bvh(objs, max_depth=20, min_objs=1):
    global MAX_LEAF_PRIMITIVES
    n_objs = len(objs)
    if n_objs == 0:
        return None

    node = BVHNode()
    for obj in objs:
        node.aabb.extend(obj.aabb)

    if n_objs <= MAX_LEAF_PRIMITIVES or max_depth <= 0:
        node.is_leaf = True
        node.primitive_indices = [o.obj_idx for o in objs]
        return node

    best_cost = float('inf')
    best_axis = 0
    best_split_pos = 0.0
    best_left_objs = []
    best_right_objs = []

    aabb_extent = node.aabb.max - node.aabb.min
    axes = [(0, aabb_extent.x), (1, aabb_extent.y), (2, aabb_extent.z)]
    axes.sort(key=lambda x: x[1], reverse=True)

    for axis, _ in axes[:2]:
        try:
            objs_sorted = sorted(objs, key=lambda o: o.centroid[axis])
        except IndexError:
            node.is_leaf = True
            node.primitive_indices = [o.obj_idx for o in objs]
            return node

        min_centroid = min(o.centroid[axis] for o in objs_sorted)
        max_centroid = max(o.centroid[axis] for o in objs_sorted)
        if min_centroid == max_centroid:
            continue

        for i in range(1, 10):
            split_pos = min_centroid + (max_centroid - min_centroid) * (i / 10.0)
            cost, left_aabb, right_aabb, left_count, right_count = compute_sah_split(objs_sorted, axis, split_pos)
            if cost < best_cost and left_count > 0 and right_count > 0:
                best_cost = cost
                best_axis = axis
                best_split_pos = split_pos
                best_left_objs = [o for o in objs_sorted if o.centroid[axis] < split_pos]
                best_right_objs = [o for o in objs_sorted if o.centroid[axis] >= split_pos]

    parent_cost = n_objs * node.aabb.surface_area()
    if best_cost >= parent_cost or not best_left_objs:
        node.is_leaf = True
        node.primitive_indices = [o.obj_idx for o in objs]
        return node

    node.left_child_node_obj = build_bvh(best_left_objs, max_depth - 1, min_objs)
    node.right_child_node_obj = build_bvh(best_right_objs, max_depth - 1, min_objs)

    if node.left_child_node_obj is None and node.right_child_node_obj is None:
        node.is_leaf = True
        node.primitive_indices = [o.obj_idx for o in objs]
        if hasattr(node, 'left_child_node_obj'):
            delattr(node, 'left_child_node_obj')
        if hasattr(node, 'right_child_node_obj'):
            delattr(node, 'right_child_node_obj')
        node.left_child_idx = -1
        node.right_child_idx = -1

    return node

def flatten_bvh_py(node_obj):
    global bvh_flat_nodes, node_counter
    if node_obj is None:
        return -1
    curr_id = node_counter
    node_counter += 1
    bvh_flat_nodes.append(node_obj)
    node_obj.node_idx = curr_id
    if not node_obj.is_leaf:
        node_obj.left_child_idx = flatten_bvh_py(node_obj.left_child_node_obj)
        node_obj.right_child_idx = flatten_bvh_py(node_obj.right_child_node_obj)
        if hasattr(node_obj, 'left_child_node_obj'):
            delattr(node_obj, 'left_child_node_obj')
        if hasattr(node_obj, 'right_child_node_obj'):
            delattr(node_obj, 'right_child_node_obj')
    return curr_id

def create_many_sdf_empties(num_to_create):
    print(f"Create/update {num_to_create} empties...")
    gs = int(np.ceil(np.sqrt(num_to_create)))
    sp = 2.0
    sts = ["sphere", "box", "torus", "cylinder"]
    c = 0
    for i in range(gs):
        if c >= num_to_create:
            break
        for j in range(gs):
            if c >= num_to_create:
                break
            n = f"SDF_Shape_{c:04d}"
            ct = random.choice(sts)
            ps = []
            if ct == "sphere":
                ps = [random.uniform(0.3, 0.7)]
            elif ct == "box":
                ps = [random.uniform(0.2, 0.5) for _ in range(3)]
            elif ct == "torus":
                ps = [random.uniform(0.4, 0.8), random.uniform(0.1, 0.3)]
            elif ct == "cylinder":
                ps = [random.uniform(0.2, 0.5), random.uniform(0.3, 0.7)]
            lx = (j - gs / 2.0) * sp
            ly = (i - gs / 2.0) * sp
            lz = random.uniform(-1, 1)
            loc = (lx, ly, lz)
            if n not in bpy.data.objects:
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=loc)
                e = bpy.context.active_object
                e.name = n
                e.empty_display_size = 0.1
            else:
                e = bpy.data.objects[n]
                e.location = loc
            e["sdf_type"] = ct
            e["sdf_params"] = ps
            e.rotation_euler = tuple(random.uniform(0, np.pi * 2) for _ in range(3))
            e.scale = tuple(random.uniform(0.7, 1.3) for _ in range(3))
            c += 1
    print(f"Done: {c} empties.")

def gen_quad_verts():
    return [(-1, -1), (1, -1), (-1, 1), (-1, 1), (1, -1), (1, 1)]

vs = '''in vec2 position;out vec2 uv_frag;void main(){uv_frag=position*0.5+0.5;gl_Position=vec4(position,0,1);}'''
fs = '''uniform vec3 cameraPos_world;uniform float maxDist;uniform bool isPerspective;
uniform mat4 vpm,inv_vpm,inv_vm;uniform vec2 viewportSize;
uniform sampler2D bvhABBTex,bvhMetaTex,shapeDataTex, bvhLeafPrimsTex; 
uniform int num_bvh_nodes; 
uniform int numActiveShapesInTex; // For direct SDF evaluation for normals
uniform float rimCenterBalance; 
uniform float rimSharpness;    
in vec2 uv_frag;out vec4 FragColor;

mat4 get_mat(int si,int btx){vec4 c0=texelFetch(shapeDataTex,ivec2(btx,si),0);vec4 c1=texelFetch(shapeDataTex,ivec2(btx+1,si),0);vec4 c2=texelFetch(shapeDataTex,ivec2(btx+2,si),0);vec4 c3=texelFetch(shapeDataTex,ivec2(btx+3,si),0);return mat4(c0,c1,c2,c3);}
float s_sph(vec3 p,mat4 wm,float r){mat4 im=inverse(wm);vec3 pl=(im*vec4(p,1.0)).xyz;vec3 s=vec3(length(wm[0].xyz),length(wm[1].xyz),length(wm[2].xyz));float ms=max(0.0001,min(s.x,min(s.y,s.z)));pl/=ms;return(length(pl)-r)*ms;}
float s_box(vec3 p,mat4 wm,vec3 b){mat4 im=inverse(wm);vec3 pl=(im*vec4(p,1.0)).xyz;vec3 s=vec3(length(wm[0].xyz),length(wm[1].xyz),length(wm[2].xyz));float ms=max(0.0001,min(s.x,min(s.y,s.z)));pl/=ms;vec3 q=abs(pl)-b;float dl=length(max(q,vec3(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);return dl*ms;}
float s_tor(vec3 p,mat4 wm,vec2 t){mat4 im=inverse(wm);vec3 pl=(im*vec4(p,1.0)).xyz;vec3 s=vec3(length(wm[0].xyz),length(wm[1].xyz),length(wm[2].xyz));float ms=max(0.0001,min(s.x,min(s.y,s.z)));pl/=ms;vec2 q=vec2(length(pl.xz)-t.x,pl.y);float dl=length(q)-t.y;return dl*ms;}
float s_cyl(vec3 p,mat4 wm,vec2 h){mat4 im=inverse(wm);vec3 pl=(im*vec4(p,1.0)).xyz;vec3 s=vec3(length(wm[0].xyz),length(wm[1].xyz),length(wm[2].xyz));float ms=max(0.0001,min(s.x,min(s.y,s.z)));pl/=ms;vec2 da=abs(vec2(length(pl.xz),pl.y))-h;float dl=min(max(da.x,da.y),0.0)+length(max(da,vec2(0.0)));return dl*ms;}
float smin(float a,float b,float k){float h=max(k-abs(a-b),0.0)/k;return min(a,b)-h*h*k*0.25;}

bool intersectRayAABB(vec3 ro, vec3 invRd, vec3 bmin, vec3 bmax, out float ten, out float tex) {
    vec3 t1=(bmin-ro)*invRd; vec3 t2=(bmax-ro)*invRd;
    vec3 tmn=min(t1,t2); vec3 tmx=max(t1,t2);
    ten=max(tmn.x,max(tmn.y,tmn.z)); tex=min(tmx.x,min(tmx.y,tmx.z));
    return ten<tex && tex>0.0;
}

float sdf_scene_direct_no_bvh(vec3 prp) {
    float msd = maxDist;
    for (int sdi = 0; sdi < numActiveShapesInTex; ++sdi) {
        vec4 pt = texelFetch(shapeDataTex, ivec2(0, sdi), 0);
        float stid = pt.x; mat4 swm = get_mat(sdi, 1); float dsh = maxDist;
        if (stid < 0.5) dsh = s_sph(prp, swm, pt.y);
        else if (stid < 1.5) dsh = s_box(prp, swm, pt.yzw);
        else if (stid < 2.5) dsh = s_tor(prp, swm, pt.yz);
        else if (stid < 3.5) dsh = s_cyl(prp, swm, pt.yz);
        msd = smin(msd, dsh, 0.3);
    } return msd;
}
vec3 calculate_finite_normal(vec3 p) {
    float eps = 0.001;
    const vec2 k = vec2(1.0, -1.0);
    return normalize( k.xyy * sdf_scene_direct_no_bvh( p + k.xyy*eps ) +
                      k.yyx * sdf_scene_direct_no_bvh( p + k.yyx*eps ) +
                      k.yxy * sdf_scene_direct_no_bvh( p + k.yxy*eps ) +
                      k.xxx * sdf_scene_direct_no_bvh( p + k.xxx*eps ) );
}

float sdf_bvh(vec3 prp, vec3 ray_origin_world, vec3 ray_direction_world, vec3 inv_ray_dir) {
    float min_scene_dist_at_prp = maxDist;
    int stack[64]; int stack_ptr = 0;
    if (num_bvh_nodes == 0) return maxDist;
    stack[stack_ptr++] = 0;
    float t_entry_aabb, t_exit_aabb;
    float t_entry_child1, t_exit_child1; float t_entry_child2, t_exit_child2;
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0 || node_idx >= num_bvh_nodes) continue;
        vec3 node_aabb_min = texelFetch(bvhABBTex, ivec2(0, node_idx), 0).xyz;
        vec3 node_aabb_max = texelFetch(bvhABBTex, ivec2(1, node_idx), 0).xyz;
        if (!intersectRayAABB(ray_origin_world, inv_ray_dir, node_aabb_min, node_aabb_max, t_entry_aabb, t_exit_aabb)) {
            continue;
        }
        vec4 node_meta = texelFetch(bvhMetaTex, ivec2(0, node_idx), 0);
        float is_leaf_flag = node_meta.a;
        if (is_leaf_flag > 0.5) {
            int leaf_prims_offset = int(node_meta.r);
            int prim_count_in_leaf = int(node_meta.g);
            float d_leaf_combined = maxDist;
            for (int i = 0; i < prim_count_in_leaf; ++i) {
                int shape_data_idx = int(texelFetch(bvhLeafPrimsTex, ivec2(0, leaf_prims_offset + i), 0).r);
                if (shape_data_idx < 0) continue;
                vec4 params_type = texelFetch(shapeDataTex, ivec2(0, shape_data_idx), 0);
                float shape_type_id = params_type.x;
                mat4 shape_world_mat = get_mat(shape_data_idx, 1);
                float d_shape = maxDist;
                if (shape_type_id < 0.5) d_shape = s_sph(prp, shape_world_mat, params_type.y);
                else if (shape_type_id < 1.5) d_shape = s_box(prp, shape_world_mat, params_type.yzw);
                else if (shape_type_id < 2.5) d_shape = s_tor(prp, shape_world_mat, params_type.yz);
                else if (shape_type_id < 3.5) d_shape = s_cyl(prp, shape_world_mat, params_type.yz);
                d_leaf_combined = smin(d_leaf_combined, d_shape, 0.1);
            }
            min_scene_dist_at_prp = smin(min_scene_dist_at_prp, d_leaf_combined, 0.3);
        } else {
            int left_child_idx = int(node_meta.r); int right_child_idx = int(node_meta.g);
            bool hit_left = false; bool hit_right = false;
            vec3 left_aabb_min, left_aabb_max, right_aabb_min, right_aabb_max;
            if (left_child_idx >= 0) {
                left_aabb_min = texelFetch(bvhABBTex, ivec2(0, left_child_idx), 0).xyz;
                left_aabb_max = texelFetch(bvhABBTex, ivec2(1, left_child_idx), 0).xyz;
                hit_left = intersectRayAABB(ray_origin_world, inv_ray_dir, left_aabb_min, left_aabb_max, t_entry_child1, t_exit_child1);
            }
            if (right_child_idx >= 0) {
                right_aabb_min = texelFetch(bvhABBTex, ivec2(0, right_child_idx), 0).xyz;
                right_aabb_max = texelFetch(bvhABBTex, ivec2(1, right_child_idx), 0).xyz;
                hit_right = intersectRayAABB(ray_origin_world, inv_ray_dir, right_aabb_min, right_aabb_max, t_entry_child2, t_exit_child2);
            }
            if (hit_left && hit_right) {
                if (t_entry_child1 < t_entry_child2) {
                    if (stack_ptr < 64) stack[stack_ptr++] = right_child_idx; if (stack_ptr < 64) stack[stack_ptr++] = left_child_idx;
                } else {
                    if (stack_ptr < 64) stack[stack_ptr++] = left_child_idx; if (stack_ptr < 64) stack[stack_ptr++] = right_child_idx;
                }
            } else if (hit_left) { if (stack_ptr < 64) stack[stack_ptr++] = left_child_idx;
            } else if (hit_right) { if (stack_ptr < 64) stack[stack_ptr++] = right_child_idx; }
        }
    } return min_scene_dist_at_prp;
}

vec3 rim_center_shade(vec3 world_normal, vec3 world_view_dir, float balance, float sharpness) {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(world_view_dir);
    float dotNV = abs(dot(N, V));
    
    float highlight_factor;
    if (dotNV < balance) {
        highlight_factor = 1.0 - (dotNV / balance);
    } else {
        highlight_factor = (dotNV - balance) / (1.0 - balance + 1e-6);
    }
    highlight_factor = clamp(highlight_factor, 0.0, 1.0);
    highlight_factor = pow(highlight_factor, sharpness);

    vec3 white_color = vec3(1.0);
    vec3 grey_color = vec3(0.3);
    return mix(grey_color, white_color, highlight_factor);
}

void main(){vec2 suv=gl_FragCoord.xy/viewportSize;vec2 ndc=suv*2.0-1.0;vec3 ro;vec3 rd;
if(isPerspective){ro=cameraPos_world;vec4 fc=vec4(ndc.x,ndc.y,1.0,1.0);vec4 fw=inv_vpm*fc;rd=normalize(fw.xyz/fw.w-ro);}
else{vec4 nc=vec4(ndc.x,ndc.y,-1.0,1.0);vec4 nw=inv_vpm*nc;ro=nw.xyz/nw.w;rd=normalize((inv_vm*vec4(0.0,0.0,-1.0,0.0)).xyz);}
vec3 inv_rd = vec3(0.0); 
if (abs(rd.x) > 1e-6) inv_rd.x = 1.0/rd.x; else inv_rd.x = sign(rd.x)*1e6; 
if (abs(rd.y) > 1e-6) inv_rd.y = 1.0/rd.y; else inv_rd.y = sign(rd.y)*1e6;
if (abs(rd.z) > 1e-6) inv_rd.z = 1.0/rd.z; else inv_rd.z = sign(rd.z)*1e6;

float t=0.0;const int MRS=512;const float HE=0.001;
for(int i=0;i<MRS;++i){
    vec3 cp=ro+t*rd; 
    float d=sdf_bvh(cp,ro,rd,inv_rd);
    if(abs(d)<HE){
        vec3 n = calculate_finite_normal(cp);
        
        vec3 view_dir = normalize(cameraPos_world - cp);
        vec3 surface_color = rim_center_shade(n, view_dir, rimCenterBalance, rimSharpness); 
        
        vec4 clp=vpm*vec4(cp,1.0);float dp=clp.z/clp.w;gl_FragDepth=(dp*0.5+0.5);
        FragColor=vec4(surface_color,1.0);return;
    }
    t+=max(d*0.8,HE*0.5);if(t>maxDist)break;
}
discard;}
'''

def upd_sh_tex(objs):
    global shape_texture, shape_texture_buffer
    nvs = 0
    dnp = np.zeros(MAX_TEXTURE_SHAPES * SHAPE_TEXTURE_WIDTH * 4, dtype=np.float32)
    tm = {"sphere": 0.0, "box": 1.0, "torus": 2.0, "cylinder": 3.0}
    for i, obj_d in enumerate(objs):
        if i >= MAX_TEXTURE_SHAPES:
            break
        tid = tm.get(obj_d.sdf_type_str, 0.0)
        ps = [tid] + obj_d.sdf_params
        pad_ps = (ps + [0.0, 0.0, 0.0])[:4]
        idx0 = (i * SHAPE_TEXTURE_WIDTH * 4)
        dnp[idx0:idx0 + 4] = pad_ps
        mat = obj_d.matrix_world.transposed()
        for col_i in range(4):
            mat_idx = (i * SHAPE_TEXTURE_WIDTH * 4) + ((1 + col_i) * 4)
            dnp[mat_idx:mat_idx + 4] = mat[col_i]
        nvs += 1
    dlist = dnp.tolist()
    blen = len(dlist)
    chg = False
    try:
        if shape_texture_buffer is None or len(shape_texture_buffer) != blen:
            shape_texture_buffer = gpu.types.Buffer('FLOAT', blen)
            chg = True
        shape_texture_buffer[:] = dlist
        if not chg:
            chg = True
    except Exception as e:
        print(f"FATAL ShapeBuf: {e}")
        return None, 0
    tdims = (SHAPE_TEXTURE_WIDTH, MAX_TEXTURE_SHAPES)
    if shape_texture is None or chg:
        try:
            shape_texture = gpu.types.GPUTexture(size=tdims, format='RGBA32F', data=shape_texture_buffer)
            if hasattr(shape_texture, 'interpolation'):
                shape_texture.interpolation = 'Closest'
            elif hasattr(shape_texture, 'use_interpolation'):
                shape_texture.use_interpolation = False
        except Exception as e:
            print(f"FATAL ShapeTex: {e}")
            return None, 0
    return shape_texture, nvs

def upd_bvh_tex(flat_bvh_node_list, all_leaf_primitive_indices_flat):
    global bvh_aabb_texture, bvh_aabb_buffer, bvh_meta_texture, bvh_meta_buffer, \
           bvh_leaf_prims_texture, bvh_leaf_prims_buffer

    if not flat_bvh_node_list:
        bvh_aabb_texture = None
        bvh_meta_texture = None
        bvh_leaf_prims_texture = None
        return False
    num_nodes = len(flat_bvh_node_list)

    aabb_w = 2
    aabb_np = np.zeros(num_nodes * aabb_w * 4, dtype=np.float32)
    for i, node in enumerate(flat_bvh_node_list):
        idx_min = (i * aabb_w * 4)
        aabb_np[idx_min:idx_min + 3] = node.aabb.min[:]
        aabb_np[idx_min + 3] = 0.0
        idx_max = (i * aabb_w * 4) + 4
        aabb_np[idx_max:idx_max + 3] = node.aabb.max[:]
        aabb_np[idx_max + 3] = 0.0
    aabb_list = aabb_np.tolist()
    aabb_buf_len = len(aabb_list)
    aabb_chg = False
    try:
        if bvh_aabb_buffer is None or len(bvh_aabb_buffer) != aabb_buf_len:
            bvh_aabb_buffer = gpu.types.Buffer('FLOAT', aabb_buf_len)
            aabb_chg = True
        bvh_aabb_buffer[:] = aabb_list
        if not aabb_chg:
            aabb_chg = True
    except Exception as e:
        print(f"FATAL BVH AABB Buffer: {e}")
        return False
    aabb_dims = (aabb_w, num_nodes)
    if bvh_aabb_texture is None or aabb_chg:
        try:
            bvh_aabb_texture = gpu.types.GPUTexture(size=aabb_dims, format='RGBA32F', data=bvh_aabb_buffer)
            if hasattr(bvh_aabb_texture, 'interpolation'):
                bvh_aabb_texture.interpolation = 'Closest'
        except Exception as e:
            print(f"FATAL BVH AABB Texture: {e}")
            return False
    
    meta_w = 1
    meta_np = np.zeros(num_nodes * meta_w * 4, dtype=np.float32)
    for i, node in enumerate(flat_bvh_node_list):
        idx = (i * meta_w * 4)
        r_val, g_val, b_val, a_val = 0.0, 0.0, 0.0, 0.0
        if node.is_leaf:
            a_val = 1.0
            r_val = float(node.leaf_prims_offset)
            g_val = float(len(node.primitive_indices))
        else:
            a_val = 0.0
            r_val = float(node.left_child_idx)
            g_val = float(node.right_child_idx)
        meta_np[idx] = r_val
        meta_np[idx + 1] = g_val
        meta_np[idx + 2] = b_val
        meta_np[idx + 3] = a_val
    meta_list = meta_np.tolist()
    meta_buf_len = len(meta_list)
    meta_chg = False
    try:
        if bvh_meta_buffer is None or len(bvh_meta_buffer) != meta_buf_len:
            bvh_meta_buffer = gpu.types.Buffer('FLOAT', meta_buf_len)
            meta_chg = True
        bvh_meta_buffer[:] = meta_list
        if not meta_chg:
            meta_chg = True
    except Exception as e:
        print(f"FATAL BVH Meta Buffer: {e}")
        return False
    meta_dims = (meta_w, num_nodes)
    if bvh_meta_texture is None or meta_chg:
        try:
            bvh_meta_texture = gpu.types.GPUTexture(size=meta_dims, format='RGBA32F', data=bvh_meta_buffer)
            if hasattr(bvh_meta_texture, 'interpolation'):
                bvh_meta_texture.interpolation = 'Closest'
        except Exception as e:
            print(f"FATAL BVH Meta Texture: {e}")
            return False

    if not all_leaf_primitive_indices_flat:
        bvh_leaf_prims_texture = None
    else:
        leaf_prims_rgba_np = np.zeros(len(all_leaf_primitive_indices_flat) * 4, dtype=np.float32)
        for i_prim_idx, actual_prim_idx in enumerate(all_leaf_primitive_indices_flat):
            leaf_prims_rgba_np[i_prim_idx * 4] = float(actual_prim_idx)
        leaf_prims_list = leaf_prims_rgba_np.tolist()
        leaf_prims_buf_len = len(leaf_prims_list)
        leaf_prims_chg = False
        try:
            if bvh_leaf_prims_buffer is None or len(bvh_leaf_prims_buffer) != leaf_prims_buf_len:
                bvh_leaf_prims_buffer = gpu.types.Buffer('FLOAT', leaf_prims_buf_len)
                leaf_prims_chg = True
            bvh_leaf_prims_buffer[:] = leaf_prims_list
            if not leaf_prims_chg:
                leaf_prims_chg = True
        except Exception as e:
            print(f"FATAL BVH Leaf Prims Buffer: {e}")
            return False
        
        texture_height = max(1, len(all_leaf_primitive_indices_flat))
        leaf_prims_tex_dims = (1, texture_height)
        if bvh_leaf_prims_texture is None or leaf_prims_chg or \
           (bvh_leaf_prims_texture and bvh_leaf_prims_texture.size[1] != texture_height):
            try:
                bvh_leaf_prims_texture = gpu.types.GPUTexture(size=leaf_prims_tex_dims, format='RGBA32F', data=bvh_leaf_prims_buffer)
                if hasattr(bvh_leaf_prims_texture, 'interpolation'):
                    bvh_leaf_prims_texture.interpolation = 'Closest'
            except Exception as e:
                print(f"FATAL BVH Leaf Prims Texture: {e}")
                return False
    return True

def draw_cb():
    global shader_object, bvh_flat_nodes, node_counter, \
           bvh_aabb_texture, bvh_meta_texture, bvh_leaf_prims_texture

    if shader_object is None:
        return
    ctx, reg, r3d = bpy.context, bpy.context.region, bpy.context.space_data.region_3d
    if not all([ctx, reg, r3d]) or reg.width == 0 or reg.height == 0:
        return
    
    sorted_blender_objects = sorted(bpy.data.objects, key=lambda o: o.name)
    all_sdf_obj_data = []
    ti = 0
    for o in sorted_blender_objects:
        if o.name.startswith("SDF_Shape_") and o.get("sdf_type") and o.get("sdf_params"):
            if ti < MAX_TEXTURE_SHAPES:
                all_sdf_obj_data.append(SDFObjectData(o, ti))
                ti += 1
    if not all_sdf_obj_data:
        return
    
    bvh_flat_nodes.clear()
    node_counter = 0
    root_node = build_bvh(list(all_sdf_obj_data))
    
    all_leaf_primitive_indices_flat = []
    if root_node:
        flatten_bvh_py(root_node)
        if not bvh_flat_nodes:
            print("Warn: BVH Flat Nodes Empty after flatten")
            return
        current_offset = 0
        for node_item in bvh_flat_nodes:
            if node_item.is_leaf:
                node_item.leaf_prims_offset = current_offset
                all_leaf_primitive_indices_flat.extend(node_item.primitive_indices)
                current_offset += len(node_item.primitive_indices)
    else:
        print("Warn: BVH Root None")
        return
    
    tex_s, num_s = upd_sh_tex(all_sdf_obj_data)
    if tex_s is None or num_s == 0:
        return
    
    if not upd_bvh_tex(bvh_flat_nodes, all_leaf_primitive_indices_flat):
        return
        
    expected_leaf_prims = len(all_leaf_primitive_indices_flat) > 0
    if bvh_aabb_texture is None or bvh_meta_texture is None or \
       (expected_leaf_prims and bvh_leaf_prims_texture is None):
        if len(all_leaf_primitive_indices_flat) > 0 and bvh_leaf_prims_texture is None:
            return
        elif bvh_aabb_texture is None or bvh_meta_texture is None:
            return

    num_bvh_nodes_val = len(bvh_flat_nodes) if bvh_flat_nodes else 0

    is_p = r3d.is_perspective
    vmat_val = r3d.view_matrix
    pmat_val = r3d.window_matrix
    vpmat_val = pmat_val @ vmat_val
    try:
        inv_vpmat_val = vpmat_val.inverted()
        inv_vmat_val = vmat_val.inverted()
        cam_pos_w_val = inv_vmat_val.translation
    except ValueError:
        return
    vpsize_val = (float(reg.width), float(reg.height))
    max_d_val = 1000.0
    
    scene = ctx.scene
    rc_balance = scene.get("sdf_rim_center_balance", DEFAULT_RIM_CENTER_BALANCE)
    rc_sharpness = scene.get("sdf_rim_sharpness", DEFAULT_RIM_SHARPNESS)

    vts = gen_quad_verts()
    bth = batch_for_shader(shader_object, 'TRIS', {"position": vts})
    shader_object.bind()
    try:
        shader_object.uniform_float("cameraPos_world", cam_pos_w_val)
        shader_object.uniform_float("maxDist", max_d_val)
        shader_object.uniform_bool("isPerspective", [is_p])
        shader_object.uniform_float("vpm", vpmat_val)
        shader_object.uniform_float("inv_vpm", inv_vpmat_val)
        shader_object.uniform_float("inv_vm", inv_vmat_val)
        shader_object.uniform_float("viewportSize", vpsize_val)
        shader_object.uniform_sampler("shapeDataTex", tex_s)
        shader_object.uniform_sampler("bvhABBTex", bvh_aabb_texture)
        shader_object.uniform_sampler("bvhMetaTex", bvh_meta_texture)
        if bvh_leaf_prims_texture:
            shader_object.uniform_sampler("bvhLeafPrimsTex", bvh_leaf_prims_texture)
        shader_object.uniform_int("num_bvh_nodes", [num_bvh_nodes_val])
        shader_object.uniform_int("numActiveShapesInTex", [num_s])
        shader_object.uniform_float("rimCenterBalance", rc_balance)
        shader_object.uniform_float("rimSharpness", rc_sharpness)
    except Exception as e:
        print(f"Uniform Bind Fail: {e}")
        return
    
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('ALPHA')
    bth.draw(shader_object)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)

# --- UI Panel for Settings ---
class SDF_PT_RaymarchSettingsPanel(bpy.types.Panel):
    bl_label = "SDF Raymarcher FX"
    bl_idname = "SDF_PT_raymarch_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "sdf_rim_center_balance")
        layout.prop(scene, "sdf_rim_sharpness")

def scene_upd_hdl(scene):
    if bpy.context.window_manager:
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == 'VIEW_3D':
                    for r_iter in a.regions:
                        if r_iter.type == 'WINDOW':
                            r_iter.tag_redraw()

def register_addon():
    global handle, shader_object, shape_texture, shape_texture_buffer, \
           bvh_aabb_texture, bvh_aabb_buffer, bvh_meta_texture, bvh_meta_buffer, \
           bvh_leaf_prims_texture, bvh_leaf_prims_buffer
    unregister_addon()
    try:
        shader_object = gpu.types.GPUShader(vs, fs)
        print("Shader compiled.")
    except Exception as e:
        print(f"FATAL Shader Compile: {e}")
        shader_object = None
        return
    
    bpy.types.Scene.sdf_rim_center_balance = bpy.props.FloatProperty(
        name="Rim/Center Balance", default=DEFAULT_RIM_CENTER_BALANCE, min=0.0, max=1.0,
        description="Midpoint of grey band (0=rim, 0.5=balanced, 1=center)")
    bpy.types.Scene.sdf_rim_sharpness = bpy.props.FloatProperty(
        name="Rim Sharpness", default=DEFAULT_RIM_SHARPNESS, min=0.1, max=10.0,
        description="Exponent for falloff. Higher is sharper transition to grey.")
    bpy.utils.register_class(SDF_PT_RaymarchSettingsPanel)

    shape_texture = None
    shape_texture_buffer = None
    bvh_aabb_texture = None
    bvh_aabb_buffer = None
    bvh_meta_texture = None
    bvh_meta_buffer = None
    bvh_leaf_prims_texture = None
    bvh_leaf_prims_buffer = None
    
    handle = bpy.types.SpaceView3D.draw_handler_add(draw_cb, (), 'WINDOW', 'POST_VIEW')
    print("Draw CB registered.")
    if scene_upd_hdl not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(scene_upd_hdl)
        print("Depsgraph Hdl registered.")

def unregister_addon():
    global handle, shader_object, shape_texture, shape_texture_buffer, \
           bvh_aabb_texture, bvh_aabb_buffer, bvh_meta_texture, bvh_meta_buffer, \
           bvh_leaf_prims_texture, bvh_leaf_prims_buffer
    if handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        handle = None
    if 'scene_upd_hdl' in globals() and scene_upd_hdl in bpy.app.handlers.depsgraph_update_post:
        try:
            bpy.app.handlers.depsgraph_update_post.remove(scene_upd_hdl)
        except ValueError:
            pass
    
    try:
        if SDF_PT_RaymarchSettingsPanel.bl_idname in bpy.types.bpy_classes_rna_dict:
            bpy.utils.unregister_class(SDF_PT_RaymarchSettingsPanel)
    except Exception as e:
        print(f"Error unregistering panel: {e}")

    if hasattr(bpy.types.Scene, "sdf_rim_center_balance"):
        del bpy.types.Scene.sdf_rim_center_balance
    if hasattr(bpy.types.Scene, "sdf_rim_sharpness"):
        del bpy.types.Scene.sdf_rim_sharpness

    shader_object = None
    shape_texture = None
    shape_texture_buffer = None
    bvh_aabb_texture = None
    bvh_aabb_buffer = None
    bvh_meta_texture = None
    bvh_meta_buffer = None
    bvh_leaf_prims_texture = None
    bvh_leaf_prims_buffer = None
    print("Handlers unregistered, GPU refs cleared.")

if __name__ == "__main__":
    unregister_addon()
    create_many_sdf_empties(TARGET_NUM_SHAPES)
    register_addon()
