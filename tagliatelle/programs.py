from moderngl_window.scene import MeshProgram
from moderngl_window.resources import programs
from moderngl_window.meta import ProgramDescription
import moderngl_window.scene.programs as mglw_progs
import moderngl_window as mglw

# Get Program vs Custom Program --> Tradeoff
# Here need to change attributes to match the built in programs
# With custom, each has different draw functions as well
def get_program(mesh):

    has_material = mesh.material
    has_normals = mesh.attributes.get("NORMAL")
    has_textures = mesh.attributes.get("TEXTURE")
    has_colors = mesh.attributes.get("COLOR")

    # Base check for material
    if has_material:    
        
        # Vertex Color Program
        if has_colors and not has_textures:
            return mglw_progs.VertexColorProgram()

        # Color Light Program 
        elif has_normals:
            return mglw_progs.ColorLightProgram()

        # Texture Program
        elif has_textures and not has_normals and not has_colors and mesh.mat_texture is not None:
            return mglw_progs.TextureProgram()

        # Texture Vertex Color Program
        elif not has_normals and has_textures and has_colors and mesh.material.mat_texture is not None:
            return mglw_progs.TextureVertexColorProgram()

        # Texture Light Color Program
        elif has_normals and has_textures and mesh.material.mat_texture is not None:
            return mglw_progs.TextureLightColorProgram()

        # Fallback Program
        else:
            return mglw_progs.FallbackProgram()
        
    else:
        # Fallback Program
        return mglw_progs.FallbackProgram()

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
                out vec4 f_color;

                void main() {
                    f_color = vec4(v_color);
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

        # if mesh.material:
        #     self.program["color"].value = tuple(mesh.material.color[0:3])
        # else:
        #     self.program["color"].value = (1.0, 1.0, 1.0)

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
                

                out vec4 f_color;

                void main() {
                    // using camera as light source
                    float light = dot(normalize(-v_position), normalize(v_normal));
                    f_color = v_color * (.25 + abs(light) * .75);
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

        # if mesh.material:
        #     self.program["color"].value = tuple(mesh.material.color[0:3])
        # else:
        #     self.program["color"].value = (1.0, 1.0, 1.0)

        mesh.vao.render(self.program, instances = self.num_instances)
    
    def apply(self, mesh):
        return self


def construct_program(context, instance: list[list[float]], attributes: dict):
    
    has_normal = "NORMAL" in attributes
    has_tangent = "TANGENT" in attributes
    has_texture = "TEXTURE" in attributes
    has_color = "COLOR" in attributes

    vert_shader = f'''
        #version 330
        in vec3 in_position;
        {"in vec3 in_normal;" if has_normal else ""}
        {"in vec3 in_tangent;" if has_tangent else ""}
        {"in vec3 in_texture;" if has_texture else ""}
        {"in vec4 in_color;" if has_color else ""}

        {"out vec3 v_normal;" if has_normal else ""}
        {"out vec3 v_tangent;" if has_tangent else ""}
        {"out vec3 v_texture;" if has_texture else ""}
        out vec4 v_color
        
        void main() {{
            gl_Position = vec4{instance[0][0], instance[0][1], instance[0][2], 1.0} * vec4{instance[3][0], instance[3][1], instance[3][2], 1.0};

            {"v_normal = in_normal;" if has_normal else ""}
            {"v_tangent = in_tangent;" if has_tangent else ""}
            {"v_texture = in_texture;" if has_texture else ""}
            {f"v_color = in_color * vec4{instance[1][0], instance[1][1], instance[1][2], instance[1][3]};" if has_color else f"v_color = vec4{instance[1][0], instance[1][1], instance[1][2], instance[1][3]};"}
        }}
    '''
    frag_shader = f'''
        #version 330
        in vec3 v_position;
        in vec4 v_color;
        {"in vec3 v_normal;" if has_normal else ""}
        {"in vec3 v_tangent;" if has_tangent else ""}
        {"in vec3 v_texture;" if has_texture else ""}
        
        out vec4 f_color;

        void main() {{
            {"float lum = clamp(dot(normalize(Light - v_position), normalize(v_normal)), 0.0, 1.0) * 0.8 + 0.2;" if has_normal else "1"}
            {"f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);" if has_texture else ""}
            f_color = v_color;
        }}
    '''
     
    return context.program(vertex_shader=vert_shader,fragment_shader=frag_shader,)