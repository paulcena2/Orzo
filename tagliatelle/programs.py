import os

import numpy as np
import moderngl
import moderngl_window as mglw

from moderngl_window.scene import MeshProgram
from PIL import Image

current_dir = os.path.dirname(__file__)


class PhongProgram(MeshProgram):
    """
    Instance Rendering Program with Phong Shading
    """
    current_camera_matrix = None
    camera_position = None

    def __init__(self, ctx, num_instances, **kwargs):
        super().__init__(program=None)
        self.num_instances = num_instances

        # Vertex Shader
        if num_instances == -1:
            vertex_path = os.path.join(current_dir, "shaders/base_vertex.glsl")
            vertex = open(vertex_path, 'r').read()
            num_instances = 1
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

        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        self.program["normalization_factor"].value = mesh.norm_factor

        # Only invert matrix / calculate camera position if camera is moved
        if list(camera_matrix) != PhongProgram.current_camera_matrix:
            camera_world = np.linalg.inv(camera_matrix)          
            PhongProgram.current_camera_matrix = list(camera_matrix)
            PhongProgram.camera_position = tuple(camera_world.m4[:3])
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
        lights = list(mesh.lights.values())
        num_lights = len(lights)

        # Trim lights down if exceeding max amount for buffer in shader
        # - smarter way to get closer ones could be implemented
        if num_lights > 8:
            num_lights = 8
            lights = lights[:8]

        self.program["num_lights"].value = num_lights
        for i, light in zip(range(num_lights), lights):
            for attr, val in light.items():
                self.program[f"lights[{i}].{attr}"].value = val
        # print(f"Light Positions: {[light.get('world_position') for light in lights]}")
        # print(f"Camera Position: {BaseProgram.camera_position}")

        # Hack to change culling for double_sided material
        if mesh.material.double_sided:
            mesh.vao.ctx.disable(moderngl.CULL_FACE)
        else:
            mesh.vao.ctx.enable(moderngl.CULL_FACE)

        mesh.vao.render(self.program, instances=self.num_instances)
    
    def apply(self, mesh):
        return self
