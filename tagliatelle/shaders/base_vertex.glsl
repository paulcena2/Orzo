#version 330
in vec3 in_position;

out vec4 v_color;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

void main() {
    gl_Position = m_proj * m_cam * m_model * vec4(in_position, 1.0);
    v_color = vec4(1.0, 1.0, 1.0, 1.0);
}