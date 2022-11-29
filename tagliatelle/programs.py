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

                out vec4 v_color;
                out vec3 v_normal;
                out vec3 v_position;
                out vec3 v_view_position;
                out mat4 view;

                void main() {

                    // Get rotation matrix from quaternion 
                    vec4 q = instance_matrix[2];
                    vec3 col1 = vec3(2 * (q[0]*q[0] + q[1]*q[1])-1, 2 * (q[1]*q[2] + q[0]*q[3]), 2 * (q[1]*q[3] - q[0]*q[2]));
                    vec3 col2 = vec3(2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[0]*q[0] + q[2]*q[2]) - 1, 2 * (q[2]*q[3] + q[0]*q[1]));
                    vec3 col3 = vec3(2 * (q[1]*q[3] + q[0]*q[2]), 2 * (q[2]*q[3] - q[0]*q[1]), 2 * (q[0]*q[0] + q[3]*q[3]) - 1);
                    mat3 rotation_matrix = mat3(col1, col2, col3);

                    view = m_cam;
                    mat4 mv = view * m_model;
                    vec4 position = mv * vec4((rotation_matrix * in_position * vec3(instance_matrix[3])) +  vec3(instance_matrix[0]), 1.0);

                    gl_Position = m_proj * position;

                    mat3 normal_matrix = transpose(inverse(mat3(mv)));
                    v_normal = normal_matrix * in_normal; // need to normalize?
                    v_position = position.xyz;
                    v_color = in_color * instance_matrix[1];

                    vec4 view_model_position = mv * position;
                    v_view_position = view_model_position.xyz;
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
                };
                
                in vec3 v_position;
                in vec3 v_normal;
                in vec4 v_color;
                in vec3 v_view_position;
                in mat4 view;

                uniform int num_lights;
                uniform LightInfo lights[8];
                uniform vec4 material_color;

                out vec4 f_color;

                void main() {

                    f_color = vec4(0.0);
                    int i = 0;
                    while (i < num_lights){
                        
                        LightInfo light = lights[i];

                        vec4 lightPosition = view * vec4(light.world_position, 1.0);
                        vec3 lightVector = lightPosition.xyz - v_view_position;
                        //vec3 lightVector = light.world_position - v_position;
                        float lightDistance = length(lightVector);
                        float falloff = 1 / (lightDistance * lightDistance); // place holder for now - need to figure out range

                        vec3 L = normalize(lightVector);
                        vec3 N = normalize(v_normal);

                        float specular = 0.0;
                        vec4 diffuse = light.color * max(0.0, dot(L, N)) * falloff; // using lambertian attenuation
                        vec3 ambient = light.ambient;
                        vec4 diffuseColor = material_color * v_color;

                        f_color += diffuseColor * (diffuse + vec4(ambient, 1.0)) + specular;
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

        if mesh.material:
            self.program["material_color"].value = tuple(mesh.material.color)
        else:
            self.program["material_color"].value = (1.0, 1.0, 1.0, 1.0)

        lights = mesh.lights
        num_lights = len(lights)
        self.program["num_lights"].value = num_lights

        # Set light values - better way to pass dict directly? getting value error cause key doesn't match program's dict
        light_attrs = ["world_position", "color", "ambient", "type"]
        for i, light in zip(range(num_lights), lights):
            for attr in range(len(light_attrs)):
                self.program[f"lights[{i}].{light_attrs[attr]}"].value = light[attr]

        mesh.vao.render(self.program, instances = self.num_instances)
    
    def apply(self, mesh):
        return self


