#version 330

struct LightInfo {
    vec3 world_position;
    vec4 color;
    vec3 ambient;
    int type;
    vec4 info;
    vec3 direction;
};

in vec3 world_position;
in vec3 normal;
in vec4 color;
in vec2 texcoord;
in vec3 view_vector;

uniform int num_lights;
uniform LightInfo lights[8];
uniform vec4 material_color;
uniform sampler2D base_texture;
uniform bool double_sided;
uniform float shininess;
uniform float spec_strength;
uniform float attention;
uniform bool ghosting;

out vec4 f_color;

void main() {

    f_color = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 tex_color = vec4(texture(base_texture, texcoord));

    vec3 V = normalize(view_vector);
    vec3 N = normalize(normal);

    // Correct Normal if double sided and needs to be flipped
    if (double_sided && dot(view_vector, normal) < 0)
        N = -N;

    int i = 0;
    while (i < num_lights){
        
        LightInfo light = lights[i];
        float intensity = light.info[0];
        float range = light.info[1];

        vec3 lightVector = light.world_position - world_position;
        float lightDistance = length(lightVector);
        vec3 L = normalize(lightVector);
        
        float falloff = 0.0;
        
        // Point Light
        if (light.type == 0)
            falloff = 1 / (1 + lightDistance * lightDistance);

        // Spot Light
        else if (light.type == 1) {
            falloff = 1 / (1.0 + lightDistance * lightDistance);
            float outer_angle = light.info[2];
            float inner_angle = light.info[3];
            float lightAngleScale = 1.0 / max(.001, cos(inner_angle) - cos(outer_angle));
            float lightAngleOffset = -cos(outer_angle) * lightAngleScale;
            float cd = dot(light.direction, L);
            float angularAttenuation = clamp((cd * lightAngleScale + lightAngleOffset), 0.0, 1.0);
            angularAttenuation *= angularAttenuation;
            falloff *= angularAttenuation;
        }

        // Directional Light
        else
            falloff = 1;
        
        // Computer diffuse
        vec4 diffuse = light.color * max(0.0, dot(L, N)) * falloff; // using lambertian attenuation

        // Compute Specular
        vec3 reflection = -reflect(L, N);
        float specularPower = pow(max(0.0, dot(V, reflection)), shininess);
        float specular = spec_strength * specularPower * falloff;

        // Get Ambient
        vec3 ambient = light.ambient;
        
        // Get diffuse color - tex_color is the issue -> black instead of white
        vec4 diffuseColor = material_color * color * tex_color;

        // Add contribution to final color
        //diffuseColor = material_color * tex_color;
        //f_color += diffuseColor;
        //f_color += ((diffuse + vec4(ambient, 1.0)) + specular) * intensity;
        f_color += (diffuseColor * (diffuse + vec4(ambient, 1.0)) + specular) * intensity * attention;
        i += 1;
    }

    // Add ghosting effect
    if (ghosting)
        f_color[3] = .5;

}