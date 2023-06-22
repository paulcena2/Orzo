#version 330

in mat4 instance_matrix;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texture;
in vec4 in_color;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;
uniform vec3 camera_position;
uniform float normalization_factor;

out vec4 color;
out vec3 normal;
out vec3 world_position;
out vec2 texcoord;
out vec3 view_vector;


vec3 quat_transform(vec4 q, vec3 v){
    return v + 2.0*cross(cross(v, q.xyz ) + q.w*v, q.xyz);
}

void main() {

    // Get rotation matrix from quaternion 
    vec4 q = instance_matrix[2];
    mat4 mv = m_cam * m_model;

    // Scale, rotate, then shift vertex
    vec4 local_position = vec4(quat_transform(q, (in_position * vec3(instance_matrix[3]))) +  vec3(instance_matrix[0]), 1.0);
    vec4 view_position = mv * local_position;

    gl_Position = m_proj * view_position;

    mat3 normal_matrix = mat3(m_model);
    normal = normalize(normal_matrix * quat_transform(q, in_normal));
    color = in_color * instance_matrix[1];
    world_position = (m_model * local_position).xyz;
    texcoord = in_texture / normalization_factor;
    view_vector = camera_position - world_position;

}