import queue
import logging
import os

import moderngl_window as mglw
import moderngl
import numpy as np
from pyrr import Quaternion
from pathlib import Path
import imgui
from imgui.integrations.pyglet import create_renderer
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import penne


current_dir = os.path.dirname(__file__)

SKYBOX_RADIUS = 1000.0

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


def get_char(cls, number):
    for attr_name, attr_value in cls.__dict__.items():
        if attr_value == number:
            return attr_name[-1]
    return None  # Return None if the number is not found in the mapping


def get_distance_to_mesh(camera_pos, mesh):
    """Get the distance from the camera to the mesh"""
    mesh_position = mesh.node.matrix_global[3, :3]
    return np.linalg.norm(camera_pos - mesh_position)


def normalize_device_coordinates(x, y, width, height):
    """Normalize click coordinates to NDC"""
    x = (2.0 * x) / width - 1.0
    y = 1.0 - (2.0 * y) / height
    return x, y


def distances_point_to_ray(points, ray_origin, ray_direction):
    """Compute distance from point to a ray in 3d space

    Useful for checking distance from instance to ray when clicking
    """

    distances = []
    for point in points:
        vec_to_point = point[:3] - ray_origin
        parallel_component = np.dot(vec_to_point, ray_direction)
        closest_point = ray_origin + parallel_component * ray_direction
        distances.append(np.linalg.norm(closest_point - point[:3]))
    return distances


def intersection(ray_direction, ray_origin, bbox_min, bbox_max, instance_positions=None):
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

        if instance_positions is not None:
            dists = distances_point_to_ray(instance_positions, ray_origin, ray_direction)
            return np.min(dists), t_near
        else:
            return 0, t_near

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
        self.camera.projection.update(near=0.1, far=1000.0)  # Range where camera will cutoff
        self.camera.mouse_sensitivity = 0.1
        self.camera.velocity = 2.0
        self.camera.zoom = 2.5
        self.camera_position = [0.0, 0.0, 0.0]
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
        self.rotating = False  # Flag for rotating entity on drag

        # Flag for rendering bounding boxes on mesh, can be toggled in GUI
        self.draw_bboxes = True

        # Set up skybox
        self.skybox_on = True
        self.skybox = mglw.geometry.sphere(radius=SKYBOX_RADIUS)
        self.skybox_program = self.load_program(os.path.join(current_dir, "shaders/sky.glsl"))
        self.skybox_texture = self.load_texture_2d("skybox.png", flip_y=False)

    def get_ray_from_click(self, x, y, world=True):

        # Get matrices
        projection = np.array(self.camera.projection.matrix)
        view = np.array(self.camera.matrix)
        inverse_projection = np.linalg.inv(projection)
        inverse_view = np.linalg.inv(view)

        # Normalized Device Coordinates
        x, y = normalize_device_coordinates(x, y, self.wnd.width, self.wnd.height)

        # Make vectors for click and release locations
        if world:  # use distance to mesh as part of ray length
            distance = get_distance_to_mesh(self.camera_position, self.selection)  # This is a rough estimate -> error down the road
            ray_clip = np.array([x, y, -1.0, distance], dtype=np.float32)

            # Reverse perspective division
            ray_clip[0:3] *= distance
        else:
            ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)

        # To Eye-Space
        ray_eye = np.matmul(ray_clip, inverse_projection)
        if not world:
            ray_eye[2], ray_eye[3] = -1.0, 0.0
            norm_factor = np.linalg.norm(ray_eye)
            ray_eye = ray_eye / norm_factor if norm_factor != 0 else ray_eye

        # To World-Space
        ray_world = np.matmul(ray_eye, inverse_view)

        # Reformat final ray
        ray = ray_world[:3]
        if not world:
            norm_factor = np.linalg.norm(ray)
            ray /= norm_factor if norm_factor != 0 else 1
        return ray

    def get_world_translations(self, x, y, x_last, y_last):
        """Get world translation from 2d mouse input"""

        # Get rays
        click_vec = self.get_ray_from_click(x_last, y_last)
        release_vec = self.get_ray_from_click(x, y)

        # Get the difference between the two vectors
        return release_vec - click_vec

    def get_world_rotation(self, x, y, x_last, y_last):

        # Get rays
        click_vec = self.get_ray_from_click(x_last, y_last, world=False)
        release_vec = self.get_ray_from_click(x, y, world=False)

        # Get axis of rotation with cross product
        axis = np.cross(click_vec, release_vec)
        # axis = np.cross(release_vec, click_vec)

        # Get angle of rotation with dot product
        # angle = np.arccos(np.dot(release_vec, click_vec))
        angle = .05

        # Create quaternion
        return Quaternion.from_axis_rotation(axis, angle)

    def key_event(self, key, action, modifiers):

        # Log for debugging events
        print(f"Key Entered: {key}, {action}, {modifiers}")

        # Pass event to gui
        self.gui.key_event(key, action, modifiers)
        # action = "ACTION_PRESS"  # This function only registers key_releases for all_new

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

            # Workaround: try passing it to unicode char after for pyglet==2.0.7
            uni_char = get_char(self.wnd.keys, key)
            self.unicode_char_entered(uni_char.lower())

        if key == keys.R:
            if action == keys.ACTION_PRESS:
                self.rotating = True
            elif action == keys.ACTION_RELEASE:
                self.rotating = False

    def mouse_position_event(self, x: int, y: int, dx, dy):

        # Log for debugging events
        # print(f"Mouse Position: {x}, {y}, {dx}, {dy}")

        # Pass event to gui
        self.gui.mouse_position_event(x, y, dx, dy)

        # Move camera if enabled
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def mouse_press_event(self, x: int, y: int, button: int):

        # Log for debugging events
        print(f"Mouse Press: {x}, {y}, {button}")

        # Pass event to gui
        self.gui.mouse_press_event(x, y, button)

        # If the mouse is over a window, don't do anything
        if imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW):
            return

        # Get 3d world ray
        self.last_click = (x, y)
        ray = self.get_ray_from_click(x, y, world=False)

        # Check meshes for intersection with bounding box
        closest_dtc = float('inf')
        closest_dtr = float('inf')
        closest_mesh = None
        for mesh in self.scene.meshes:
            # Convert bounding box to world space - Pad out to vec4, then transform by entity transform
            entity = self.client.get_delegate(mesh.entity_id)
            bbox_min = np.array([*mesh.bbox_min, 1.0])
            bbox_min = np.matmul(bbox_min, mesh.transform)
            bbox_max = np.array([*mesh.bbox_max, 1.0])
            bbox_max = np.matmul(bbox_max, mesh.transform)
            if entity.instance_positions is not None:
                instance_positions = np.array([np.matmul(pos, mesh.transform) for pos in entity.instance_positions])
                hit = intersection(ray, self.camera_position, bbox_min, bbox_max, instance_positions)
            else:
                hit = intersection(ray, self.camera_position, bbox_min, bbox_max)

            if not hit:  # No intersection with bounding box at all
                continue

            dist_to_ray, dist_to_camera = hit

            # Closest if ray is closer than previous closest, or if ray is same distance but closer to camera
            if dist_to_ray < closest_dtr or (dist_to_ray == closest_dtr and dist_to_camera < closest_dtc):
                closest_dtr = dist_to_ray
                closest_dtc = dist_to_camera
                closest_mesh = entity

        # Set selection in window state
        if closest_mesh:
            self.selection = closest_mesh
            logging.info(f"Clicked Mesh: {closest_mesh}")
        else:
            self.selection = None

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Change appearance by changing the mesh's transform"""

        # Log for debugging events
        print(f"Mouse Drag: {x}, {y}, {dx}, {dy}")

        # Pass event to gui
        self.gui.mouse_drag_event(x, y, dx, dy)

        if not self.selection or imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW):
            return

        x_last, y_last = x - dx, y - dy
        current_mat = self.selection.node.matrix

        if dx == 0 and dy == 0:
            return

        # If r is held, rotate, if not translate
        if self.rotating:
            quat = self.get_world_rotation(x, y, x_last, y_last)
            pos = current_mat[3, :3]  # This seems like a hack, but gets rid of the translation component
            new_mat = np.matmul(current_mat, quat.matrix44)
            new_mat[3, :3] = pos
            self.selection.node.matrix = new_mat
            self.selection.node.matrix_global = new_mat
            self.selection.node.mesh.transform = new_mat
            self.selection.node.children[0].matrix_global = new_mat
            # Transorm is in node matrix, matrix global, children -> patch's node transform, mesh transform
            # There is a mesh in the child node for the patch and also a mesh at the entity level -> double version
        else:
            dx, dy, dz = self.get_world_translations(x, y, x_last, y_last)
            translation_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, dz, 1]])
            self.selection.node.matrix = np.matmul(current_mat, translation_mat)
            self.selection.node.matrix_global = self.selection.node.matrix

    def mouse_release_event(self, x: int, y: int, button: int):
        """On release, officially send request to move the object"""

        # Log for debugging events
        print(f"Mouse Release: {x}, {y}, {button}")

        # Pass event to gui
        self.gui.mouse_release_event(x, y, button)

        # Calculate vectors and move if applicable
        if self.selection and self.last_click != (x, y):

            try:
                if self.rotating:
                    quat = Quaternion.from_matrix(self.selection.node.matrix_global)
                    self.selection.set_rotation(quat.xyzw.tolist())
                else:
                    x, y, z = self.selection.node.matrix_global[3, :3].astype(float)
                    self.selection.set_position([x, y, z])
                    #self.selection.request_move(dx, dy, dz)
            except AttributeError as e:
                logging.warning(f"Dragging {self.selection} failed: {e}")

    def resize(self, width: int, height: int):
        self.gui.resize(width, height)
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def unicode_char_entered(self, char):

        # Log for debugging events
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

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)  # was raising problem in IDE but seemed to work
        # self.ctx.enable_only(moderngl.CULL_FACE)

        # Render skybox
        if self.skybox_on:
            self.ctx.front_face = 'cw'
            self.skybox_texture.use()
            self.skybox_program['m_proj'].write(self.camera.projection.matrix)
            self.skybox_program['m_cam'].write(self.camera.matrix)
            self.skybox.render(self.skybox_program)
            self.ctx.front_face = 'ccw'

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
            logging.info(f"Callback in render: {callback.__name__} \n\tw/ args: {args}")
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

        # Main Menu
        if imgui.begin_main_menu_bar():

            if imgui.begin_menu("State", True):
                for id_type in penne.id_map.values():

                    expanded, visible = imgui.collapsing_header(f"{SPECIFIER_MAP[id_type]}", visible=True)
                    if not expanded:
                        continue

                    select_components = [component for id, component in state.items() if type(id) is id_type]
                    for delegate in select_components:
                        delegate.gui_rep()
                imgui.end_menu()
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item("Quit", 'Cmd+Q', False, True)

                if clicked_quit:
                    exit(1)  # Need to hook this up to window
                imgui.end_menu()
            if imgui.begin_menu("Settings", True):

                # Bboxes
                clicked, self.draw_bboxes = imgui.checkbox("Show Bounding Boxes", self.draw_bboxes)

                # Skybox
                clicked, self.skybox_on = imgui.checkbox("Use Skybox", self.skybox_on)

                # Camera Settings
                imgui.menu_item("Camera Settings", None, False, True)
                changed, speed = imgui.slider_float("Speed", self.camera.velocity, 0.0, 10.0, format="%.0f")
                if changed:
                    self.camera.velocity = speed

                changed, sensitivity = imgui.slider_float("Sensitivity", self.camera.mouse_sensitivity, 0.0, 1.0)
                if changed:
                    self.camera.mouse_sensitivity = sensitivity

                # Shader Settings
                imgui.menu_item("Shader Settings", None, False, True)
                shininess = self.shininess
                spec = self.spec_strength
                changed, shininess = imgui.slider_float("Shininess", shininess, 0.0, 100.0, format="%.0f",
                                                        power=1.0)
                if changed:
                    self.shininess = shininess

                changed, spec = imgui.slider_float("Specular Strength", spec, 0.0, 1.0, power=1.0)
                if changed:
                    self.spec_strength = spec

                imgui.end_menu()
            imgui.end_main_menu_bar()

        # Scene Info
        imgui.begin("Basic Info")
        imgui.text(f"Camera Position: {self.camera_position}")
        imgui.text(f"Press 'Space' to toggle camera/GUI")
        imgui.text(f"Click and drag an entity to move it")
        imgui.text(f"Hold 'r' while dragging to rotate an entity")
        _, self.draw_bboxes = imgui.checkbox("Show Bounding Boxes", self.draw_bboxes)
        imgui.end()

        # Render Document Methods and Signals or selection
        if self.selection is None:
            self.render_document()
        else:
            imgui.begin("Selection")
            self.selection.gui_rep()
            imgui.end()
