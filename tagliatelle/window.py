import moderngl_window as mglw
from moderngl_window.conf import settings
import moderngl
import numpy as np
from pathlib import Path
import queue
import json

import imgui
from imgui.integrations.pyglet import create_renderer

DEFAULT_SHININESS = 10.0
DEFAULT_SPEC_STRENGTH = 0.2

HINT_MAP = {
    "noo::any": (imgui.core.input_text, ("Any", 256), ("")),
    "noo::text": (imgui.core.input_text, ("Text", 256), ("")),
    "noo::integer": (imgui.core.input_int, ("Int"), ("")),
    "noo::real": (imgui.core.input_float, ("Real"), (1.0)),
    "noo::array": (imgui.core.input_text, ["[Array]", 256], ["[]"]),
    "noo::map": (imgui.core.input_text, ["{dict}", "{}", 256]),
    "noo::any_id": (imgui.core.input_int2, ["Id", 0, 0]),
    "noo::entity_id": (imgui.core.input_int2, ["Entity Id"], [0, 0]),
    "noo::table_id": (imgui.core.input_int2, ["Table Id", 0, 0]),
    "noo::plot_id": (imgui.core.input_int2, ["Plot Id", 0, 0]),
    "noo::method_id": (imgui.core.input_int2, ["Method Id", 0, 0]),
    "noo::signal_id": (imgui.core.input_int2, ["Signal Id", 0, 0]),
    "noo::image_id": (imgui.core.input_int2, ["Image Id", 0, 0]),
    "noo::sampler_id": (imgui.core.input_int2, ["Sampler Id", 0, 0]),
    "noo::texture_id": (imgui.core.input_int2, ["Texture Id", 0, 0]),
    "noo::material_id": (imgui.core.input_int2, ["Material Id", 0, 0]),
    "noo::light_id": (imgui.core.input_int2, ["Light Id", 0, 0]),
    "noo::buffer_id": (imgui.core.input_int2, ["Buffer Id", 0, 0]),
    "noo::bufferview_id": (imgui.core.input_int2, ["Buffer View Id", 0, 0]),
    "noo::range(a,b,c)": (imgui.core.input_float3, ["Range (a->b) step by c", 0, 0, 0]),
}

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
        super().__init__(**kwargs)

        # Set Up Camera
        self.camera = mglw.scene.camera.KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.1
        self.camera.velocity = 1.0
        self.camera.zoom = 2.5
        self.camera_position = None
        
        # Window Options
        self.wnd.mouse_exclusivity = True
        self.camera_enabled = True
        
        # Store Light Info
        self.lights = {} # light_id: light_info

        # Create scene and set up basic nodes
        self.scene = mglw.scene.Scene("Noodles Scene")
        root = mglw.scene.Node("Root")
        root.matrix_global = np.identity(4, np.float32)
        self.scene.root_nodes.append(root)
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
        self.update_gui()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

        try:
            callback_info = Window.client.callback_queue.get(block=False)
            callback, args = callback_info
            print(f"Callback in render: {callback} \n\tw/ args: {args}")
            callback(self, *args)

        except queue.Empty:
            pass


    def update_gui(self):

        imgui.new_frame()
        state = self.client.state

        # Main Menu
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(1) # Need to hook this up to window

                imgui.end_menu()

            imgui.end_main_menu_bar()

        # State Inspector
        imgui.begin("State")
        for specifier in state:

            expanded, visible = imgui.collapsing_header(specifier, visible=True)
            if not expanded or specifier == "document":
                continue

            for id, delegate in state[specifier].items():
                delegate.gui_rep()
        imgui.end()

        # Scene Info
        imgui.begin("Basic Info")
        imgui.text(f"Camera Position: {self.camera_position}")
        imgui.text(f"Press 'Space' to toggle camera/GUI")
        imgui.end()

        # Methods
        self.render_methods()

        # Shader Settings
        shininess = self.shininess
        spec = self.spec_strength
        imgui.begin("Shader")
        changed, shininess = imgui.slider_float("Shininess", shininess, 0.0, 100.0, format="%.0f", power=1)
        if changed:
            self.shininess = shininess

        changed, spec = imgui.slider_float("Specular Strength", spec, 0.0, 1.0, power=1)
        if changed:
            self.spec_strength = spec
        imgui.end()


    def render_methods(self):
        # Methods
        imgui.begin("Methods")
        state = self.client.state
        for id, delegate in state["methods"].items():
            imgui.begin_group()
            
            if imgui.button(f"Invoke {delegate.name}"):
                if delegate.info.arg_doc == []:
                    self.client.invoke_method(delegate.name, [])
                else:
                    imgui.open_popup(f"Invoke {id}")
            imgui.core.push_text_wrap_pos()
            imgui.text(f"Docs: {delegate.docs}")
            imgui.core.pop_text_wrap_pos()
            imgui.separator()


            if imgui.begin_popup(f"Invoke {id}"):

                imgui.text("Input Arguments")
                imgui.separator()
                for arg in delegate.info.arg_doc:
                    imgui.text(arg.name.upper())
                    
                    # Get input block from state or get default
                    if arg.name in self.args:
                        component, parameters, vals = self.args[arg.name]
                    else:
                        try:
                            hint = arg.editor_hint if hasattr(arg, "editor_hint") else "noo::any"
                            component, parameters, vals = HINT_MAP[hint]
                        except:
                            raise Exception(f"Invalid Hint for {arg.name} arg")

                    label, rest = parameters[0], parameters[1:]
                    if isinstance(vals, list):
                        changed, values = component(label, *vals, *rest)
                    else:
                        changed, values = component(f"{label} for {arg.name}", vals, *rest)

                    self.args[arg.name] = (component, parameters, values)
                    imgui.text(arg.doc)
                    imgui.separator()
                
                if imgui.button("Submit"):

                    # Get vals and convert type if applicable
                    final_vals = []
                    for arg in self.args.values():
                        value = arg[2]
                        try:
                            clean_val = json.loads(value)
                        except:
                            clean_val = value
                        final_vals.append(clean_val)

                    print(f"Invoking the method: {delegate.name} w/ args: {final_vals}")
                    self.client.invoke_method(delegate.name, final_vals)
                imgui.end_popup()
            imgui.end_group()

        imgui.end()