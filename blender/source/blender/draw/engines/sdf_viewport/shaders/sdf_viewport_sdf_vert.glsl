void main() {
    v_ndc = position;
    gl_Position = float4(position, -1.0, 1.0);
}