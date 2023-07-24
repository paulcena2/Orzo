#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_cam;
uniform mat4 m_proj;

out vec3 pos;

void main() {

    // Get rid of translation so that the skybox is always centered on the camera
    mat4 new_cam = m_cam;
    new_cam[3][0] = 0.0;
    new_cam[3][1] = 0.0;
    new_cam[3][2] = 0.0;

    gl_Position = m_proj * new_cam * vec4(in_position, 1.0);;
    pos = in_position.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 frag_color;

uniform sampler2D skybox_texture;

in vec3 pos;

void main() {

    // Calculate the normalized direction vector from the fragment position
    vec3 view_vector = normalize(pos);

    // Convert the direction vector to polar coordinates (latitude and longitude)
    float longitude = atan(view_vector.z, view_vector.x);
    float latitude = acos(view_vector.y);

    // Normalize the polar coordinates to UV coordinates
    float u = (longitude + 3.14159265359) / (2.0 * 3.14159265359);
    float v = latitude / 3.14159265359;

    // Fetch the color from the skybox texture using the UV coordinates
    vec3 skybox_color = texture(skybox_texture, vec2(u, v)).rgb;

    // Set the fragment color
    frag_color = vec4(skybox_color, 1.0);

}
#endif
