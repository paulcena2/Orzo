#version 330
in vec4 v_color;

uniform vec4 material_color;

out vec4 f_color;

void main() {
    f_color = vec4(v_color) * material_color;
}