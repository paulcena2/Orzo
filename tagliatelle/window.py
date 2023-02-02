import moderngl_window as mglw
from moderngl_window.conf import settings
import moderngl
import numpy as np
from pathlib import Path
import queue

import imgui
from imgui.integrations.pyglet import PygletRenderer
import pyglet


class GUI:

    def __init__(self, window):
        
        imgui.create_context()

        pyglet_wnd = window.wnd._window
        self.renderer = PygletRenderer(pyglet_wnd)

        imgui.get_io().display_size = 100, 100
        imgui.get_io().fonts.get_tex_data_as_rgba32()
        
        # imgui.new_frame()  
        # imgui.end_frame()

        # Window variables
        self.test_input = 0

    def render(self):
        
       
        imgui.new_frame()

        imgui.begin("Test Window")
        imgui.text("This is the test window.")
        changed, self.test_input = imgui.input_int("Integer Input Test", self.test_input)

        imgui.end()

        self.renderer.render(imgui.get_draw_data())
        imgui.end_frame()


class Window(mglw.WindowConfig):
    """Base Window with built in 3D camera support    
    
    Most work happens in the render function which is called every frame
    """
  
    gl_version = (3, 3)
    aspect_ratio = 16 / 9
    resource_dir = Path(__file__).parent.resolve() / 'resources/'
    title = "Tagliatelle Window"
    resizable = True
    client = None
    #vsync=True
    # MATCH SCREEN RATE

    def __init__(self, **kwargs):
        #settings.WINDOW['class'] = 'moderngl_window.context.pyqt5.window'
        super().__init__(**kwargs)
        
        # self.wnd = mglw.create_window_from_settings()
        # self.ctx = self.wnd.ctx

        # Set Up Camera
        self.camera = mglw.scene.camera.KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.1
        self.camera.velocity = 1.0
        self.camera.zoom = 2.5
        
        # Window Options
        self.wnd.mouse_exclusivity = True
        self.camera_enabled = True
        
        # Store Light Info
        self.lights = {} # light_id: light_info
        self.num_lights = 0

        # Create scene and set up basic nodes
        self.scene = mglw.scene.Scene("Noodles Scene")
        root = mglw.scene.Node("Root")
        root.matrix_global = np.identity(4, np.float32)
        self.scene.root_nodes.append(root)
        self.scene.cameras.append(self.camera)

        # Set up GUI
        #self.GUI = GUI(self)
        window = self.wnd._window
        self.batch = pyglet.graphics.Batch()
        pyglet.text.Label('Hello, world', font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center', batch=self.batch)

        pyglet.gui.TextEntry(text="Enter", x=0, y=window.height//2, width=200, text_color=(255,255,255,255), batch=self.batch, )

    def key_event(self, key, action, modifiers):

        keys = self.wnd.keys
        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C or key == keys.SPACE:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.P:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def render(self, time: float, frametime: float):
        """Renders a frame to on the window
        
        Most work done in the draw function which draws each node in the scene.
        When drawing each node, the mesh is drawn, using the mesh program.
        At each frame, the callback_queue is checked so the client can update the render
        Note: each callback has the window as the first arg
        """

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        # Render GUI elements
        self.batch.draw()

        try:
            callback_info = Window.client.callback_queue.get(block=False)
            callback, args = callback_info
            print(f"Callback in render: {callback} \n\tw/ args: {args}")
            callback(self, *args)

        except queue.Empty:
            pass

