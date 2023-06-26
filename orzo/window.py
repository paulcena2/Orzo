import queue
import logging

import moderngl_window as mglw
import moderngl
import numpy as np
from pathlib import Path
import imgui
from imgui.integrations.pyglet import create_renderer

import penne


DEFAULT_SHININESS = 10.0
DEFAULT_SPEC_STRENGTH = 0.2

SPECIFIER_MAP = {
    penne.MethodID: "Methods",
    penne.SignalID: "Signals",
    penne.TableID: "Tables",
    penne.PlotID: "Plots",
    penne.EntityID: "Entities",
    penne.MaterialID: "Materials",
    penne.GeometryID: "Geometries",
    penne.LightID: "Lights",
    penne.ImageID: "Images",
    penne.TextureID: "Textures",
    penne.SamplerID: "Samplers",
    penne.BufferID: "Buffers",
    penne.BufferViewID: "Buffer Views",
    None: "Document"
}


class Window(mglw.WindowConfig):
    """Base Window with built-in 3D camera support
    
    Most work happens in the render function which is called every frame
    """

    gl_version = (3, 3)
    aspect_ratio = 16 / 9
    resource_dir = Path(__file__).parent.resolve() / 'resources/'
    title = "Orzo Window"
    resizable = True
    client = None

    # vsync=True
    # MATCH SCREEN RATE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set Up Camera
        self.camera = mglw.scene.camera.KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.1
        self.camera.velocity = 1.0
        self.camera.zoom = 2.5
        self.camera_position = None
        # self.key_repeat = True

        # Window Options
        self.wnd.mouse_exclusivity = True
        self.camera_enabled = True

        # Store Light Info
        self.lights = {}  # light_id: light_info
        # self.default_lighting = False

        # Create scene and set up basic nodes
        self.scene = mglw.scene.Scene("Noodles Scene")
        self.root = mglw.scene.Node("Root")
        self.root.matrix_global = np.identity(4, np.float32)
        self.scene.root_nodes.append(self.root)
        self.scene.cameras.append(self.camera)

        # Store shader settings
        self.shininess = DEFAULT_SHININESS
        self.spec_strength = DEFAULT_SPEC_STRENGTH

        # Set up GUI
        imgui.create_context()
        self.impl = create_renderer(self.wnd._window)
        self.args = {}

    def key_event(self, key, action, modifiers):

        keys = self.wnd.keys
        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:  # it looks like this is broken for later versions of pyglet
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
        self.update_gui()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

        try:
            callback_info = Window.client.callback_queue.get(block=False)
            callback, args = callback_info
            logging.info(f"Callback in render: {callback} \n\tw/ args: {args}")
            callback(self, *args)

        except queue.Empty:
            pass

    def update_gui(self):

        imgui.new_frame()
        state = self.client.state

        # Shader Settings
        shininess = self.shininess
        spec = self.spec_strength
        imgui.begin("Shader")
        changed, shininess = imgui.slider_float("Shininess", shininess, 0.0, 100.0, format="%.0f", power=1.0)
        if changed:
            self.shininess = shininess

        changed, spec = imgui.slider_float("Specular Strength", spec, 0.0, 1.0, power=1.0)
        if changed:
            self.spec_strength = spec
        imgui.end()

        # Main Menu
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item("Quit", 'Cmd+Q', False, True)

                if clicked_quit:
                    exit(1)  # Need to hook this up to window
                imgui.end_menu()
            imgui.end_main_menu_bar()

        # State Inspector
        imgui.begin("State")
        for id_type in penne.id_map.values():

            expanded, visible = imgui.collapsing_header(f"{SPECIFIER_MAP[id_type]}", visible=True)
            if not expanded:
                continue

            select_components = [component for id, component in state.items() if type(id) is id_type]
            for delegate in select_components:
                delegate.gui_rep()
        imgui.end()

        # Scene Info
        imgui.begin("Basic Info")
        imgui.text(f"Camera Position: {self.camera_position}")
        imgui.text(f"Press 'Space' to toggle camera/GUI")
        imgui.end()

        # Render Document Methods and Signals
        imgui.begin("Document")
        document = state["document"]
        imgui.text(f"Methods")
        imgui.separator()
        for method_id in document.methods_list:
            method = self.client.get_component(method_id)
            method.invoke_rep()

        imgui.text(f"Signals")
        imgui.separator()
        for signal_id in document.signals_list:
            signal = self.client.get_component(signal_id)
            signal.gui_rep()
        imgui.end()
