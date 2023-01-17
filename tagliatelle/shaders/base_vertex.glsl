#version 330
in vec3 in_position;

in vec3 in_normal;
in vec2 in_texture;
in vec4 in_color;

out vec4 color;
out vec3 normal;
out vec3 world_position;
out vec2 texcoord;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

void main() {
    gl_Position = m_proj * m_cam * m_model * vec4(in_position, 1.0);
    
    color = vec4(1.0, 1.0, 1.0, 1.0);
    normal = in_normal;
    world_position = gl_Position.xyz;
    texcoord = in_texture;
}