#version 330
// https://github.com/moderngl/moderngl-window/blob/master/examples/resources/programs/cubemap.glsl

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_cam;
uniform mat4 m_proj;

out vec3 pos;

void main() {
    gl_Position =  m_proj * m_cam * vec4(in_position, 1.0);;
    pos = in_position.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

uniform samplerCube texture0;

in vec3 pos;

void main() {
    fragColor = texture(texture0, normalize(pos));
}
#endif