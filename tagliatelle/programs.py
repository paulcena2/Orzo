import numpy as np

from moderngl_window.scene import MeshProgram


class BaseProgram(MeshProgram):
    """
    Default Program, for now only uses
    """

    def __init__(self, ctx, **kwargs):
        super().__init__(program=None)
        self.program = ctx.program(
            vertex_shader='''
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
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;

                uniform vec4 material_color;

                out vec4 f_color;

                void main() {
                    f_color = vec4(v_color) * material_color;
                }
            ''',
        )

    def draw(
        self,
        mesh,
        projection_matrix=None,
        model_matrix=None,
        camera_matrix=None,
        time=0,
    ):

        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)

        if mesh.material:
            self.program["material_color"].value = tuple(mesh.material.color)
        else:
            self.program["material_color"].value = (1.0, 1.0, 1.0, 1.0)

        mesh.vao.render(self.program)
    
    def apply(self, mesh):
        return self


class InstanceProgram(MeshProgram):
    """
    Instance Rendering Program
    """

    def __init__(self, ctx, num_instances, **kwargs):
        super().__init__(program=None)
        self.num_instances = num_instances
        self.program = ctx.program(
            vertex_shader='''
                #version 330

                in vec3 in_position;
                in mat4 instance_matrix;

                in vec3 in_normal;
                in vec2 in_texture;
                in vec4 in_color;
                
                uniform mat4 m_proj;
                uniform mat4 m_model;
                uniform mat4 m_cam;

                out vec4 v_color;
                out vec3 v_normal;
                out vec3 v_position;

                void main() {

                    // Get rotation matrix from quaternion 
                    vec4 q = instance_matrix[2];
                    vec3 col1 = vec3(2 * (q[0]*q[0] + q[1]*q[1])-1, 2 * (q[1]*q[2] + q[0]*q[3]), 2 * (q[1]*q[3] - q[0]*q[2]));
                    vec3 col2 = vec3(2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[0]*q[0] + q[2]*q[2]) - 1, 2 * (q[2]*q[3] + q[0]*q[1]));
                    vec3 col3 = vec3(2 * (q[1]*q[3] + q[0]*q[2]), 2 * (q[2]*q[3] - q[0]*q[1]), 2 * (q[0]*q[0] + q[3]*q[3]) - 1);
                    mat3 rotation_matrix = mat3(col1, col2, col3);

                    mat4 mv = m_cam * m_model;
                    vec4 position = mv * vec4((rotation_matrix * in_position * vec3(instance_matrix[3])) +  vec3(instance_matrix[0]), 1.0);

                    gl_Position = m_proj * position;

                    mat3 normal_matrix = transpose(inverse(mat3(mv)));
                    v_normal = normal_matrix * in_normal;
                    v_position = position.xyz;
                    v_color = in_color * instance_matrix[1];
                }
            ''',
            fragment_shader='''
                #version 330
                
                in vec3 v_position;
                in vec3 v_normal;
                in vec4 v_color;

                uniform vec4 material_color;

                out vec4 f_color;

                void main() {
                    // using camera as light source
                    float light = dot(normalize(-v_position), normalize(v_normal));
                    f_color = material_color * v_color * (.25 + abs(light) * .75);
                }
            ''',
        )

    def draw(
        self,
        mesh,
        projection_matrix=None,
        model_matrix=None,
        camera_matrix=None,
        time=0,
    ):

        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)

        if mesh.material:
            self.program["material_color"].value = tuple(mesh.material.color)
        else:
            self.program["material_color"].value = (1.0, 1.0, 1.0, 1.0)

        mesh.vao.render(self.program, instances = self.num_instances)
    
    def apply(self, mesh):
        return self


class PhongProgram(MeshProgram):
    """
    Instance Rendering Program with Phong Shading
    """

    def __init__(self, ctx, num_instances, **kwargs):
        super().__init__(program=None)
        self.num_instances = num_instances
        self.program = ctx.program(
            vertex_shader='''
                #version 330

                in vec3 in_position;
                in mat4 instance_matrix;

                in vec3 in_normal;
                in vec2 in_texture;
                in vec4 in_color;
                
                uniform mat4 m_proj;
                uniform mat4 m_model;
                uniform mat4 m_cam;

                out vec4 color;
                out vec3 normal;
                out vec3 world_position;

                void main() {

                    // Get rotation matrix from quaternion 
                    vec4 q = instance_matrix[2];
                    vec3 col1 = vec3(2 * (q[0]*q[0] + q[1]*q[1])-1, 2 * (q[1]*q[2] + q[0]*q[3]), 2 * (q[1]*q[3] - q[0]*q[2]));
                    vec3 col2 = vec3(2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[0]*q[0] + q[2]*q[2]) - 1, 2 * (q[2]*q[3] + q[0]*q[1]));
                    vec3 col3 = vec3(2 * (q[1]*q[3] + q[0]*q[2]), 2 * (q[2]*q[3] - q[0]*q[1]), 2 * (q[0]*q[0] + q[3]*q[3]) - 1);
                    mat3 rotation_matrix = mat3(col1, col2, col3);

                    mat4 mv = m_cam * m_model;
                    vec4 local_position = vec4(rotation_matrix * (in_position * vec3(instance_matrix[3])) +  vec3(instance_matrix[0]), 1.0);
                    //vec4 local_position = vec4((in_position * vec3(instance_matrix[3])) +  vec3(instance_matrix[0]), 1.0);                    
                    vec4 view_position = mv * local_position;

                    gl_Position = m_proj * view_position;

                    mat3 normal_matrix = mat3(m_model);
                    //normal = (m_model * vec4(rotation_matrix * normalize(in_normal), 1.0)).xyz;
                    //normal = (m_model * vec4(in_normal, 1.0)).xyz;
                    normal = normalize(normal_matrix * rotation_matrix * in_normal);
                    color = in_color * instance_matrix[1];
                    //color = instance_matrix[1];
                    world_position = (m_model * local_position).xyz;
                }
            ''',
            fragment_shader='''
                #version 330
                //#extension GL_OES_standard_derivatives : enable

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

                uniform int num_lights;
                uniform LightInfo lights[8];
                uniform vec4 material_color;
                uniform vec3 camera_position;

                out vec4 f_color;

                void main() {

                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    int i = 0;
                    while (i < num_lights){
                        
                        LightInfo light = lights[i];
                        float intensity = light.info[0];
                        float range = light.info[1];

                        vec3 lightVector = light.world_position - world_position;
                        float lightDistance = length(lightVector);
                        
                        vec3 L = normalize(lightVector);
                        vec3 V = normalize(camera_position - world_position);
                        vec3 N = normalize(normal);
                        
                        float falloff = 0.0;
                        // Point Light
                        if (light.type == 0)
                            falloff = 1 / (1 + lightDistance * lightDistance);
                            //falloff = 5;

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
                            falloff = 1.0;
                        
                        // Computer diffuse
                        vec4 diffuse = light.color * max(0.0, dot(L, N)) * falloff; // using lambertian attenuation

                        // Compute Specular
                        float shininess = 15.0;
                        float specularStrength = 0.5;
                        vec3 reflection = -reflect(L, N);
                        float specularPower = pow(max(0.0, dot(V, reflection)), shininess);
                        //vec3 h = normalize(V + L);
                        //float specularPower = pow(max(0.0, dot(N, h)), shininess);
                        float specular = specularStrength * specularPower * falloff;

                        // Get Ambient
                        vec3 ambient = light.ambient;
                        
                        // Get diffuse color
                        vec4 ccc = material_color;
                        vec4 diffuseColor = color;
                        f_color += diffuseColor * (diffuse + vec4(ambient, 1.0)) + specular;
                        //f_color += (vec4(ambient, 1.0)) + specular;
                        //f_color += diffuse;
                        //f_color += diffuseColor * diffuse;
                        i += 1;
                    } 
                }
            ''',
        )

    def draw(
        self,
        mesh,
        projection_matrix=None,
        model_matrix=None,
        camera_matrix=None,
        time=0,
    ):

        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)

        camera_world = np.linalg.inv(camera_matrix).m4
        #camera_position = (camera_world[2], camera_world[0], camera_world[1])
        camera_position = (camera_world[0], camera_world[1], camera_world[2])
        self.program["camera_position"].value = camera_position
        print(f"Camera Position: {camera_position}")

        if mesh.material:
            self.program["material_color"].value = tuple(mesh.material.color)
        else:
            self.program["material_color"].value = (1.0, 1.0, 1.0, 1.0)

        lights = mesh.lights.values()
        num_lights = len(lights)
        self.program["num_lights"].value = num_lights

        # Set light values - better way to pass dict directly?
        for i, light in zip(range(num_lights), lights):
            for attr, val in light.items():
                self.program[f"lights[{i}].{attr}"].value = val
                #self.program[f"lights[{i}].{attr}"].write(bytes(val))
       
        # positions = [light.get("world_position") for light in lights]
        # print(f"Light Positions: {positions}")
        mesh.vao.render(self.program, instances = self.num_instances)
    
    def apply(self, mesh):
        return self


