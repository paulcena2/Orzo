import os

import numpy as np
import moderngl
import pyrr.matrix44 as m44

from moderngl_window.scene import MeshProgram
from moderngl_window.geometry import sphere
from PIL import Image

current_dir = os.path.dirname(__file__)


class PhongProgram(MeshProgram):
    """Instance Rendering Program with Phong Shading

    Passes all necessary information to shaders and applies affects like highlighting and ghosting.
    Draws bounding sphere if enabled. Also inputs the default lighting.
    """
    current_camera_matrix = None
    camera_position = None

    def __init__(self, wnd, num_instances, **kwargs):
        super().__init__(program=None)
        self.window = wnd
        ctx = wnd.ctx
        self.num_instances = num_instances

        # Bounding box shader if used
        self.bs_program = wnd.load_program(os.path.join(current_dir, "shaders/bounding_sphere.glsl"))

        # Vertex Shader
        if num_instances == -1:
            vertex_path = os.path.join(current_dir, "shaders/base_vertex.glsl")
            vertex = open(vertex_path, 'r').read()
        else:
            vertex_path = os.path.join(current_dir, "shaders/instance_vertex.glsl")
            vertex = open(vertex_path, 'r').read()

        # Fragment Shader
        fragment_path = os.path.join(current_dir, "shaders/phong_fragment.glsl")
        fragment = open(fragment_path, 'r').read()

        self.program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)

        # Set up default texture
        img = Image.open(os.path.join(current_dir, "resources/default.png"))
        texture = ctx.texture(img.size, 4, img.tobytes())
        texture.repeat_x, texture.repeat_y = False, False
        self.default_texture = texture

    def draw(
        self,
        mesh,
        projection_matrix=None,
        model_matrix=None,
        camera_matrix=None,
        time=0,
    ):

        model_matrix = model_matrix.astype(np.float32, order='C')
        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        self.program["shininess"].value = self.window.shininess
        self.program["spec_strength"].value = self.window.spec_strength
        self.program["ghosting"].value = mesh.ghosting

        # Draw bounding box if enabled
        if self.window.draw_bs and mesh.has_bounding_sphere:
            center, radius = mesh.bounding_sphere

            # Set up matrix to shift bounding sphere to center
            shifted_model_matrix = np.identity(4, dtype='f4')
            shifted_model_matrix[3, :3] = center
            self.bs_program["m_proj"].write(projection_matrix)
            self.bs_program["m_model"].write(shifted_model_matrix)
            self.bs_program["m_cam"].write(camera_matrix)

            # Make the VAO and render
            bs_vao = sphere(radius)
            bs_vao.render(self.bs_program, mode=moderngl.LINES)

        # Add highlight effect if there is a selection, everything not selected gets a little dull
        selection = self.window.selected_entity
        if selection is not None and selection.id != mesh.entity_id:
            self.program["attention"].value = 0.5
        else:
            self.program["attention"].value = 1.0

        # Only invert matrix / calculate camera position if camera is moved
        if list(camera_matrix) != PhongProgram.current_camera_matrix:
            camera_world = np.linalg.inv(camera_matrix)          
            PhongProgram.current_camera_matrix = list(camera_matrix)
            PhongProgram.camera_position = tuple(camera_world.m4[:3])
            self.window.camera_position = [round(x, 2) for x in PhongProgram.camera_position]
        self.program["camera_position"].value = PhongProgram.camera_position

        # Feed Material in if present
        if mesh.material:
            self.program["material_color"].value = tuple(mesh.material.color)
            self.program["double_sided"].value = mesh.material.double_sided
            if mesh.material.mat_texture:
                mesh.material.mat_texture.texture.use()
            else:
                self.default_texture.use()
        else:
            self.program["material_color"].value = (1.0, 1.0, 1.0, 1.0)
            self.program["double_sided"].value = False
            self.default_texture.use()

        # Set light values
        lights = list(self.window.lights.values())
        num_lights = len(lights)

        # Default lighting
        if self.window.default_lighting:
            default_sun = {
                "world_position": (0, 500, 1000),
                "color": (1, 1, 1, 1),
                "ambient": (.5, .5, .5),
                "type": 2,
                "info": (.9, -1, 0, 0),
                "direction": (0, 0, 0)
            }
            second = {
                "world_position": (0, 0, 4),
                "color": (1, 1, 1, 1),
                "ambient": (.1, .1, .1),
                "type": 0,
                "info": (1, -1, 0, 0),
                "direction": (0, 0, 0)
            }
            lights.append(default_sun)
            lights.append(second)
            num_lights += 2

        # Trim lights down if exceeding max amount for buffer in shader
        # - smarter way to get closer ones could be implemented
        if num_lights > 8:
            num_lights = 8
            lights = lights[:8]

        self.program["num_lights"].value = num_lights
        for i, light in zip(range(num_lights), lights):
            for attr, val in light.items():
                self.program[f"lights[{i}].{attr}"].value = val

        # Hack to change culling for double_sided material
        if mesh.material.double_sided:
            mesh.vao.ctx.disable(moderngl.CULL_FACE)
        else:
            mesh.vao.ctx.enable(moderngl.CULL_FACE)

        num_instances = 1 if self.num_instances == -1 else self.num_instances
        mesh.vao.render(self.program, instances=num_instances)
    
    def apply(self, mesh):
        return self


class FrameSelectProgram(MeshProgram):
    """Render scene to frame buffer for selection

    Really simple render but instead of 'color' there is a vec4 of the following format
    [entity_id_slot, entity_id_gen, instance_number, hit_value]
    """

    current_camera_matrix = None
    camera_position = None

    def __init__(self, wnd, num_instances, **kwargs):
        super().__init__(program=None)
        self.window = wnd
        ctx = wnd.ctx
        self.num_instances = num_instances

        # Vertex Shader
        if num_instances == -1:
            vertex_path = os.path.join(current_dir, "shaders/base_vertex.glsl")
            vertex = open(vertex_path, 'r').read()
        else:
            vertex_path = os.path.join(current_dir, "shaders/instance_vertex.glsl")
            vertex = open(vertex_path, 'r').read()

        # Fragment Shader
        fragment_path = os.path.join(current_dir, "shaders/select_fragment.glsl")
        fragment = open(fragment_path, 'r').read()

        self.program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)

    def draw(
            self,
            mesh,
            projection_matrix=None,
            model_matrix=None,
            camera_matrix=None,
            time=0,
    ):

        model_matrix = model_matrix.astype(np.float32, order='C')
        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        self.program["id"].value = tuple(mesh.entity_id)

        # Set flag for widget or actual entity - hit value is zero anywhere there is no mesh
        if mesh.name == "noo::widget_cone":
            self.program["hit_value"].value = 2
        elif mesh.name == "noo::widget_torus":
            self.program["hit_value"].value = 3
        elif mesh.name == "noo::widget_tab":
            self.program["hit_value"].value = 4
        else:
            self.program["hit_value"].value = 1

        # Only invert matrix / calculate camera position if camera is moved
        if list(camera_matrix) != FrameSelectProgram.current_camera_matrix:
            camera_world = np.linalg.inv(camera_matrix)
            FrameSelectProgram.current_camera_matrix = list(camera_matrix)
            FrameSelectProgram.camera_position = tuple(camera_world.m4[:3])
        self.program["camera_position"].value = FrameSelectProgram.camera_position

        # Hack to change culling for double_sided material
        if hasattr(mesh.material, "double_sided") and mesh.material.double_sided:
            mesh.vao.ctx.disable(moderngl.CULL_FACE)
        else:
            mesh.vao.ctx.enable(moderngl.CULL_FACE)
        mesh.vao.ctx.enable(moderngl.DEPTH_TEST)

        num_instances = 1 if self.num_instances == -1 else self.num_instances
        mesh.vao.render(self.program, instances=num_instances)

    def apply(self, mesh):
        return self
