import queue
import logging

import moderngl_window as mglw
import moderngl
import numpy as np
from pathlib import Path
import imgui
from imgui.integrations.pyglet import create_renderer
from moderngl_window.integrations.imgui import ModernglWindowRenderer

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


def intersection(ray_direction, ray_origin, bbox_min, bbox_max):
    """Ray-BoundingBox intersection test"""
    t_near = float('-inf')
    t_far = float('inf')

    for i in range(3):
        if ray_direction[i] == 0:
            if ray_origin[i] < bbox_min[i] or ray_origin[i] > bbox_max[i]:
                return False
        else:
            t1 = (bbox_min[i] - ray_origin[i]) / ray_direction[i]
            t2 = (bbox_max[i] - ray_origin[i]) / ray_direction[i]
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))

    # if there is an intersection, return the distance
    if t_near <= t_far:
        return t_near
    return False


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
        # self.gui = create_renderer(self.wnd._window)
        self.gui = ModernglWindowRenderer(self.wnd)
        self.args = {}
        self.selection = None  # Current entity that is selected
        self.last_click = None  # (x, y) of last click

        # Flag for rendering bounding boxes on mesh, can be toggled in GUI
        self.draw_bboxes = True

        # Set up skybox
        # program = self.ctx.program(
        #     vertex_shader='''
        #         #version 330
        #         uniform mat4 projection;
        #         uniform mat4 view;
        #         in vec3 in_vert;
        #         out vec3 frag_coord;
        #         void main() {
        #             gl_Position = projection * view * vec4(in_vert, 1.0);
        #             frag_coord = in_vert;
        #         }
        #     ''',
        #     fragment_shader='''
        #         #version 330
        #         in vec3 frag_coord;
        #         out vec4 fragColor;
        #         uniform samplerCube skybox;
        #         void main() {
        #             fragColor = texture(skybox, normalize(frag_coord));
        #         }
        #     '''
        # )
        #
        # # Create cube geometry
        # vertices = np.array([
        #     # Cube vertices
        #     -1.0, -1.0, -1.0,
        #     1.0, -1.0, -1.0,
        #     -1.0, 1.0, -1.0,
        #     1.0, 1.0, -1.0,
        #     # ... (define the remaining vertices for the cube)
        # ], dtype='f4')
        #
        # # Upload vertex data to GPU buffer
        # vbo = self.ctx.buffer(vertices)
        # vao = self.ctx.simple_vertex_array(program, vbo, 'in_vert')
        #
        # # Load and bind the skybox textures
        # texture_filenames = [
        #     'right.jpg',
        #     'left.jpg',
        #     'top.jpg',
        #     'bottom.jpg',
        #     'front.jpg',
        #     'back.jpg'
        # ]
        # textures = []
        # for filename in texture_filenames:
        #     texture = self.ctx.texture((1024, 1024), 3)  # Replace with the appropriate texture size
        #     texture.load(filename)  # Load the texture image
        #     textures.append(texture)
        #
        # with textures[0], textures[1], textures[2], textures[3], textures[4], textures[5]:
        #     texture_unit = 0
        #     for texture in textures:
        #         texture.use(texture_unit)
        #         texture_unit += 1
        #
        # # Set shader uniforms (projection and view matrices)
        #
        # # Render the cube
        # program['skybox'].value = 0  # Bind texture unit 0
        # vao.render(moderngl.TRIANGLE_STRIP)

    def key_event(self, key, action, modifiers):

        # Pass event to gui
        self.gui.key_event(key, action, modifiers)
        # action = "ACTION_PRESS"  # This function only registers key_releases rn

        # Move camera if enabled
        keys = self.wnd.keys
        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        # Handle key presses like quit and toggle camera
        if action == keys.ACTION_PRESS:  # it looks like this is broken for later versions of pyglet
            if key == keys.C or key == keys.SPACE:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.P:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):

        # Pass event to gui
        self.gui.mouse_position_event(x, y, dx, dy)

        # Move camera if enabled
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def mouse_press_event(self, x: int, y: int, button: int):

        # Pass event to gui
        self.gui.mouse_press_event(x, y, button)

        # Convert to normalized device coordinates
        x = (2.0 * x) / self.wnd.width - 1.0
        y = 1.0 - (2.0 * y) / self.wnd.height
        self.last_click = (x, y)
        print(f"Mouse press: {x}, {y}")

        # If the mouse is over a window, don't do anything
        if imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW):
            return

        # Get matrices
        projection = np.array(self.camera.projection.matrix)
        view = np.array(self.camera.matrix)
        inverse_projection = np.linalg.inv(projection)
        inverse_view = np.linalg.inv(view)

        # Device space -> clip space -> eye space -> world space
        ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)
        ray_eye = np.matmul(ray_clip, inverse_projection)  # Pre-multiplying col-major = post-multiplying row-major
        ray_eye[2], ray_eye[3] = -1.0, 0.0
        norm_factor = np.linalg.norm(ray_eye)
        ray_eye = ray_eye / norm_factor if norm_factor != 0 else ray_eye
        ray_world = np.matmul(ray_eye, inverse_view)

        # Reformat final ray and normalize
        ray = ray_world[:3]
        norm_factor = np.linalg.norm(ray)
        ray = ray / norm_factor if norm_factor != 0 else ray

        # Check meshes for intersection with bounding box
        closest = float('inf')
        closest_mesh = None
        for mesh in self.scene.meshes:
            # Convert bounding box to world space - Pad out to vec4, then transform by entity transform
            bbox_min = np.array([*mesh.bbox_min, 1.0])
            bbox_min = np.matmul(bbox_min, mesh.transform)
            bbox_max = np.array([*mesh.bbox_max, 1.0])
            bbox_max = np.matmul(bbox_max, mesh.transform)
            hit = intersection(ray, self.camera_position, bbox_min, bbox_max)
            if hit and hit < closest:
                closest = hit
                closest_mesh = mesh

        # Set selection in window state
        if closest_mesh:
            self.selection = self.client.get_delegate(closest_mesh.entity_id)
            logging.info(f"Clicked Mesh: {closest_mesh} - {closest_mesh.entity_id}")
        else:
            self.selection = None

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Change appearance by changing the mesh's transform"""

        # Pass event to gui
        self.gui.mouse_drag_event(x, y, dx, dy)

        print(f"Dragging: {x}, {y}, {dx}, {dy}")
        return

        # if not self.selection:
        #     return
        #
        # current_mat = self.selection.node.matrix
        # translation_mat = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.selection.node.matrix = np.matmul(current_mat, translation_mat)
        #
        # print(f"New Matrix: {self.selection.node.matrix}")
        # print()

    def mouse_release_event(self, x: int, y: int, button: int):
        """On release, officially send request to move the object"""

        # Pass event to gui
        self.gui.mouse_release_event(x, y, button)

        x = (2.0 * x) / self.wnd.width - 1.0
        y = 1.0 - (2.0 * y) / self.wnd.height
        x_last, y_last = self.last_click
        dx = x - x_last
        dy = y - y_last
        print(f"∆x: {dx}, ∆y: {dy}")
        if self.selection and self.last_click != (x, y):
            try:
                self.selection.request_move(dx, dy)
            except AttributeError:
                logging.warning(f"Dragging {self.selection} failed")

    def resize(self, width: int, height: int):
        self.gui.resize(width, height)
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def unicode_char_entered(self, char):
        print(f"Unicode char entered: {char}")

        # Pass event to gui
        self.gui.unicode_char_entered(char)

        # For current versions of dependencies, key presses are not being registered and handled
        # As a workaround, we can send unicode_char_entered to the keypress event handler
        # Having this problem with mglw==2.4.4, pyglet==2.0.8, imgui==2.0.0
        # upper_char = char.upper()
        # key_num = self.wnd.keys[upper_char]
        # modifiers = mglw.context.base.keys.KeyModifiers()
        # self.key_event(key_num, 'ACTION_PRESS', modifiers)

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
        self.gui.render(imgui.get_draw_data())

        try:
            callback_info = Window.client.callback_queue.get(block=False)
            callback, args = callback_info
            logging.info(f"Callback in render: {callback} \n\tw/ args: {args}")
            callback(self, *args)

        except queue.Empty:
            pass

    def render_document(self):
        imgui.begin("Document")
        document = self.client.state["document"]

        # Methods
        imgui.text(f"Methods")
        imgui.separator()
        for method_id in document.methods_list:
            method = self.client.get_delegate(method_id)
            method.invoke_rep()

        # Signals
        imgui.text(f"Signals")
        imgui.separator()
        for signal_id in document.signals_list:
            signal = self.client.get_delegate(signal_id)
            signal.gui_rep()
        imgui.end()

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
        _, self.draw_bboxes = imgui.checkbox("Show Bounding Boxes", self.draw_bboxes)
        imgui.end()

        # Render Document Methods and Signals or selection
        if self.selection is None:
            self.render_document()
        else:
            imgui.begin("Selection")
            self.selection.gui_rep()
            imgui.end()
