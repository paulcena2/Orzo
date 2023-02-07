import moderngl_window as mglw
from moderngl_window.conf import settings
import moderngl
import numpy as np
from pathlib import Path
import queue
import pyglet

import imgui
from imgui.integrations.pyglet import create_renderer


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
        imgui.create_context()
        self.impl = create_renderer(self.wnd._window)

        # Pyglet Gui
        window = self.wnd._window
        self.batch = pyglet.graphics.Batch()
        default = pyglet.font.load(("Proggy Clean", "Times New Roman"), 13)
        pyglet.text.Label("Press 'Space' to toggle camera/GUI", font_name=default,
                          font_size=12,
                          x=window.width-140, y=20,
                          anchor_x='center', anchor_y='center', batch=self.batch)


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

            if expanded and specifier != "document":
                for id, delegate in state[specifier].items():
                    imgui.text(f"{id}: {delegate}")
        imgui.end()

        # Scene Info
        imgui.begin("Scene Info")
        imgui.text(f"Camera Position: {None}")
        imgui.text(f"Light Positions: {None}")
        imgui.end()

        # Methods
        imgui.begin("Methods")
        for id, delegate in state["methods"].items():
            imgui.begin_group()
            
            if imgui.button(f"Invoke {delegate.name}"):
                if delegate.info.arg_doc == []:
                    self.client.invoke_method(delegate.name, [])
                else:
                    imgui.open_popup(f"Invoke {id}")
            imgui.text(f"Docs: {delegate.docs}")
            imgui.separator()


            if imgui.begin_popup(f"Invoke {id}"):

                imgui.text("Input Arguments")
                imgui.separator()
                args = {}
                for arg in delegate.info.arg_doc:
                    imgui.text(arg.name.upper())
                    changed, values = imgui.input_int4(f"({arg.editor_hint}", 1,1,1,1)
                    args[arg.name] = values
                    imgui.text(arg.doc)
                    imgui.separator()
                
                if imgui.button("Submit"):
                    print(f"Invoking the method: {delegate.name} w/ args: {args}")
                    self.client.invoke_method(delegate.name, list(args.values()))
                imgui.end_popup()
            imgui.end_group()

        imgui.end()


