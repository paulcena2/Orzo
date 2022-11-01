from moderngl_window.scene import MeshProgram
from moderngl_window.resources import programs
from moderngl_window.meta import ProgramDescription
import moderngl_window.scene.programs as mglw_progs
import moderngl_window as mglw

# Rough Draft Currently Unused
class FlexMeshProgram(MeshProgram):
    """Vertex color program"""

    def __init__(self, program=None, **kwargs):
        super().__init__(program=None)
        self.program = None
        # self.program = programs.load(
        #     ProgramDescription(path="scene_default/vertex_color.glsl")
        # )

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
        mesh.vao.render(self.program)

    def apply(self, mesh):

        has_material = mesh.material
        has_normals = mesh.attributes.get("NORMAL")
        has_textures = mesh.attributes.get("TEXTURE")
        has_colors = mesh.attributes.get("COLOR")

        # Base check for material
        if has_material:    
            
            # Vertex Color Program
            if has_colors and not has_textures:
                self.program = self.program = programs.load(
                    ProgramDescription(path="scene_default/vertex_color.glsl")
                )
            # Color Light Program - 
            elif has_normals:
                self.program = self.program = programs.load(
                    ProgramDescription(path="scene_default/color_light.glsl")
                )
            # Texture Program -
            elif has_textures and not has_normals and not has_colors and mesh.mat_texture is not None:
                self.program = programs.load(
                    ProgramDescription(path="scene_default/texture.glsl")
                )
            # Texture Vertex Color Program
            # elif not has_normals and has_textures and has_colors and mesh.material.mat_texture is not None
            


        else:
            return None

        return self


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


class InstanceProgram(MeshProgram):
    """
    Default Program, for now assume we are getting just vertices
    """

    def __init__(self, ctx, instance_buffer, **kwargs):
        super().__init__(program=None)
        self.instance_buffer = instance_buffer
        self.program = ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_pos;
                in float in_scale;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = vec4(in_pos + (in_vert * in_scale), 0.0, 1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
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
            self.program["color"].value = tuple(mesh.material.color[0:3])
        else:
            self.program["color"].value = (1.0, 1.0, 1.0)

        mesh.vao.render(self.program, instances=len())
    
    def apply(self, mesh):
        return self


def construct_program(instance: list[list[float]]):
    # TODO
    return None

