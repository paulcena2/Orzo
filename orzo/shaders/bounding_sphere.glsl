#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

void main() {
    gl_Position = m_proj * m_cam * m_model * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

void main()
{
    fragColor = vec4(1.0, 1.0, 1.0, 0.5);
}

#endif