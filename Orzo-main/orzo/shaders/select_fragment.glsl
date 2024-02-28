#version 330
// Used for GPU accelerated selection
// The 'color' or the frame stores the entity id, instance, and a hit value
// The hit value indicates -1 for widget hit, 0 for no hit, and 1 for entity hit

in vec3 world_position;
in vec3 normal;
in vec4 color;
in vec2 texcoord;
in vec3 view_vector;
in float instance_id;

uniform vec2 id;
uniform int hit_value;

out vec4 f_color;

void main() {

    int instance = int(instance_id);
    f_color = ivec4(id, instance, hit_value);

}