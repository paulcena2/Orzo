#version 330

in vec3 world_position;
in vec3 normal;
in vec4 color;
in vec2 texcoord;
in vec3 view_vector;
in float instance_id;

uniform vec2 id;

out vec4 f_color;

void main() {

    int instance = int(instance_id);
    f_color = vec4(id, instance, 1.0);

}