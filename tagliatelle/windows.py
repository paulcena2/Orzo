import moderngl_window as mglw
import moderngl
import numpy as np
from pathlib import Path
import queue

class GridWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)
    title = "Test Window"
    aspect_ratio = 16 / 9
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Do initialization here
        print(type(self.ctx))
        # self.prog = self.ctx.program(...) # create program in moderngl context for shaders
        # self.vao = self.ctx.vertex_array(...)
        # self.texture = self.ctx.texture(self.wnd.size, 4)

        # START COPIED PROG - dont understand
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 vert;
                uniform float z_near;
                uniform float z_far;
                uniform float fovy;
                uniform float ratio;
                uniform vec3 center;
                uniform vec3 eye;
                uniform vec3 up;
                mat4 perspective() {
                    float zmul = (-2.0 * z_near * z_far) / (z_far - z_near);
                    float ymul = 1.0 / tan(fovy * 3.14159265 / 360);
                    float xmul = ymul / ratio;
                    return mat4(
                        xmul, 0.0, 0.0, 0.0,
                        0.0, ymul, 0.0, 0.0,
                        0.0, 0.0, -1.0, -1.0,
                        0.0, 0.0, zmul, 0.0
                    );
                }
                mat4 lookat() {
                    vec3 forward = normalize(center - eye);
                    vec3 side = normalize(cross(forward, up));
                    vec3 upward = cross(side, forward);
                    return mat4(
                        side.x, upward.x, -forward.x, 0,
                        side.y, upward.y, -forward.y, 0,
                        side.z, upward.z, -forward.z, 0,
                        -dot(eye, side), -dot(eye, upward), dot(eye, forward), 1
                    );
                }
                void main() {
                    gl_Position = perspective() * lookat() * vec4(vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 color;
                void main() {
                    color = vec4(0.04, 0.04, 0.04, 1.0);
                }
            ''',
        )

        self.prog['z_near'].value = 0.1
        self.prog['z_far'].value = 1000.0
        self.prog['ratio'].value = self.aspect_ratio
        self.prog['fovy'].value = 60

        self.prog['eye'].value = (3, 3, 3)
        self.prog['center'].value = (0, 0, 0)
        self.prog['up'].value = (0, 0, 1)
        # END COPIED PROG - dont understand 
        
        grid = []
        for i in range(65):
            grid.append([i - 32, -32.0, 0.0, i - 32, 32.0, 0.0])
            grid.append([-32.0, i - 32, 0.0, 32.0, i - 32, 0.0])

        grid = np.array(grid, dtype='f4')       

        self.vbo = self.ctx.buffer(grid)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'vert')

        self.mouse_pos = 0, 0
        camera = mglw.scene.KeyboardCamera(
            self.wnd.keys,
            fov=75.0,
            aspect_ratio=self.wnd.aspect_ratio,
            near=0.1,
            far=1000.0,
        )



    def render(self, time: float, frametime: float):
        # This method is called every frame
        # self.vao.render()
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render(moderngl.LINES, 65 * 4)
        pass

    def mouse_position_event(self, x, y, dx, dy):
        self.mouse_pos = self.mouse_pos[0] + dx, self.mouse_pos[1] + dy
        print(self.mouse_pos)

    def key_event(self, key, action, modifiers):
        print(key, action, modifiers)

    def resize(self, width: int, height: int):
        print("Window was resized. buffer size is {} x {}".format(width, height))

    def close(self):
        print("The window is closing")

class CameraWindow(mglw.WindowConfig):
    aspect_ratio = 16 / 9
    resource_dir = Path(__file__).parent.resolve() / 'resources/'
    title = "Crate.obj Model - Orbit Camera"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = mglw.scene.camera.OrbitCamera(aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True

        # self.wnd.mouse_exclusivity = True

        self.scene = self.load_scene('crate.obj')

        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 2.5

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.camera_enabled:
            self.camera.zoom_state(y_offset)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)



class Window(mglw.WindowConfig):
    """Base class with built in 3D camera support"""
  
    gl_version = (3, 3)
    aspect_ratio = 16 / 9
    resource_dir = Path(__file__).parent.resolve() / 'resources/'
    title = "Tagliatelle Window"
    resizable = True
    client = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = mglw.scene.camera.KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.1
        self.camera.zoom = 2.5
        self.wnd.mouse_exclusivity = True
        self.camera_enabled = True

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 vert;
                uniform float z_near;
                uniform float z_far;
                uniform float fovy;
                uniform float ratio;
                uniform vec3 center;
                uniform vec3 eye;
                uniform vec3 up;
                mat4 perspective() {
                    float zmul = (-2.0 * z_near * z_far) / (z_far - z_near);
                    float ymul = 1.0 / tan(fovy * 3.14159265 / 360);
                    float xmul = ymul / ratio;
                    return mat4(
                        xmul, 0.0, 0.0, 0.0,
                        0.0, ymul, 0.0, 0.0,
                        0.0, 0.0, -1.0, -1.0,
                        0.0, 0.0, zmul, 0.0
                    );
                }
                mat4 lookat() {
                    vec3 forward = normalize(center - eye);
                    vec3 side = normalize(cross(forward, up));
                    vec3 upward = cross(side, forward);
                    return mat4(
                        side.x, upward.x, -forward.x, 0,
                        side.y, upward.y, -forward.y, 0,
                        side.z, upward.z, -forward.z, 0,
                        -dot(eye, side), -dot(eye, upward), dot(eye, forward), 1
                    );
                }
                void main() {
                    gl_Position = perspective() * lookat() * vec4(vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 color;
                void main() {
                    color = vec4(0.04, 0.04, 0.04, 1.0);
                }
            ''',
        )

        self.prog['z_near'].value = 0.1
        self.prog['z_far'].value = 1000.0
        self.prog['ratio'].value = self.aspect_ratio
        self.prog['fovy'].value = 60
        self.prog['eye'].value = (3, 3, 3)
        self.prog['center'].value = (0, 0, 0)
        self.prog['up'].value = (0, 0, 1)

        grid = []
        for i in range(65):
            grid.append([i - 32, -32.0, 0.0, i - 32, 32.0, 0.0])
            grid.append([-32.0, i - 32, 0.0, 32.0, i - 32, 0.0])
        grid = np.array(grid, dtype='f4')       

        self.vbo = self.ctx.buffer(grid)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'vert')

        # self.scene = self.load_scene("crate.obj")
        self.scene = mglw.scene.Scene("Noodles Scene")
        self.scene.nodes.append(mglw.scene.Node("Root"))

    def key_event(self, key, action, modifiers):
        print(f"Key Event: {key}")
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        try:
            callback_info = Window.client.callback_queue.get(block=False)
            callback, args = callback_info
            print(f"Callback in render: {callback} \n\tw/ args: {args}")
            callback(self, *args)
        except queue.Empty:
            pass

        # self.ctx.clear(1.0, 1.0, 1.0)
        # self.vao.render(moderngl.LINES, 65 * 4)
