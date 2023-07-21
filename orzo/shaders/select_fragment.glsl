#version 330

in vec3 world_position;
in vec3 normal;
in vec4 color;
in vec2 texcoord;
in vec3 view_vector;

uniform vec2 id;

out vec4 f_color;

void main() {

    f_color = vec4(id, 1.0, 1.0);

}