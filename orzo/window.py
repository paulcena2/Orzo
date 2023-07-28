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
from PIL import Image

from orzo import programs


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

        # Set up Framebuffer - used for selection
        self.fbo = self.ctx.simple_framebuffer((self.wnd.width, self.wnd.height), dtype='u4')

        # Window Options
        self.wnd.mouse_exclusivity = True
        self.camera_enabled = True

        # Store Light Info
        self.lights = {}  # light_id: light_info
        self.default_lighting = True

        # Create scene and set up basic nodes
        self.scene = mglw.scene.Scene("Noodles Scene")
        self.root = mglw.scene.Node("Root")
        self.root.matrix = np.identity(4, np.float32)
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
        self.selected_entity = None  # Current entity that is selected
        self.selected_instance = None  # Number instance that is selected
        self.last_click = None  # (x, y) of last click
        self.rotating = False  # Flag for rotating entity on drag
        self.widgeting = False  # Flag for moving using widgets on drag
        self.widget_mode = "Global"  # Whether widgets should be axis aligned or local
        self.move_widgets = True  # Flags for enabling and disabling widgets
        self.rotate_widgets = True
        self.scale_widgets = True

        # Flag for rendering bounding spheres on mesh, can be toggled in GUI
        self.draw_bs = False

        # Set up skybox
        self.skybox_on = True
        self.skybox = mglw.geometry.sphere(radius=SKYBOX_RADIUS)
        self.skybox_program = self.load_program(os.path.join(current_dir, "shaders/sky.glsl"))
        self.skybox_texture = self.load_texture_2d("skybox.png", flip_y=False)

    def update_matrices(self):
        """Update global matrices for all nodes in the scene"""
        self.root.calc_model_mat(np.identity(4))

    def add_node(self, node, parent=None):
        """Add a node to the scene

        Adds to root by default, otherwise adds to parent node
        """
        self.scene.nodes.append(node)

        # Keep track of mesh
        if node.mesh is not None:
            self.scene.meshes.append(node.mesh)

        # Attach to parent node
        if parent is None:
            self.root.add_child(node)
        else:
            parent.add_child(node)

        # update global matrices
        self.update_matrices()

    def remove_node(self, node, parent=None):
        """Remove a node from the scene"""
        self.scene.nodes.remove(node)

        # Keep track of mesh
        if node.mesh is not None:
            self.scene.meshes.remove(node.mesh)

        # Take care of parent connection
        if parent is None:
            self.root.children.remove(node)
        else:
            parent.children.remove(node)

        # Recurse on children
        for child in node.children:
            self.remove_node(child, parent=node)

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
            distance = get_distance_to_mesh(self.camera_position, self.selected_entity)  # This is a rough estimate -> error down the road
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
        # # axis = np.cross(release_vec, click_vec)
        #
        # # Get angle of rotation with dot product
        # # angle = np.arccos(np.dot(release_vec, click_vec))
        angle = .05

        # Create quaternion
        return Quaternion.from_axis_rotation(axis, angle)

        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

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

        # Rotation modifier
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

        # Get info from framebuffer and click coordinates, cast to ints
        pixel_data = self.render_scene_to_framebuffer(x, y)
        slot, gen, instance, hit = pixel_data[:4], pixel_data[4:8], pixel_data[8:12], pixel_data[12:]
        slot = int(np.frombuffer(slot, dtype=np.single))  # Why are these floats? I'm not sure but it works
        gen = int(np.frombuffer(gen, dtype=np.single))
        instance = int(np.frombuffer(instance, dtype=np.single))
        hit = int(np.frombuffer(hit, dtype=np.single))

        # No hit -> No selection
        if hit == 0:
            self.selected_entity = None
            self.selected_instance = None
            if self.widgeting:
                self.widgeting = False
                self.remove_widgets()
            return

        if hit == 2:
            self.widgeting = True
            print("Widgeting")
        else:
            self.widgeting = False

        # We hit something! -> Get selection from slot and gen in buffer
        entity_id = penne.EntityID(slot=slot, gen=gen)
        clicked_entity = self.client.get_delegate(entity_id)
        logging.info(f"Clicked: {clicked_entity}, Instance: {instance}")
        self.selected_instance = instance
        if clicked_entity is not self.selected_entity:
            if self.selected_entity is not None:
                self.remove_widgets()
            self.selected_entity = self.client.get_delegate(entity_id)
            self.add_widgets()

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Change appearance by changing the mesh's transform"""

        # Log for debugging events
        print(f"Mouse Drag: {x}, {y}, {dx}, {dy}")

        # Pass event to gui
        self.gui.mouse_drag_event(x, y, dx, dy)

        if not self.selected_entity or imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW) or (dx == 0 and dy == 0):
            return

        x_last, y_last = x - dx, y - dy
        selected_node = self.selected_entity.node
        current_mat_local = selected_node.matrix
        current_mat_global = selected_node.matrix_global

        # Turn on ghosting effect
        selected_node.mesh.ghosting = True

        # If widgeting, move using the widget's rules
        if self.widgeting:
            dx, dy, dz = self.get_world_translations(x, y, x_last, y_last)
            if self.selected_instance == 0:  # x axis
                translation_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, 0, 0, 1]])
            elif self.selected_instance == 1:  # y axis
                translation_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, dy, 0, 1]])
            elif self.selected_instance == 2:  # z axis
                translation_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, dz, 1]])
            else:
                translation_mat = np.identity(4)
            selected_node.matrix_global = np.matmul(current_mat_global, translation_mat)

        # If r is held, rotate, if not translate
        elif self.rotating:
            # quat = self.get_world_rotation(x, y, x_last, y_last)
            # new_mat = np.matmul(current_mat_global, quat.matrix44)
            # new_mat[3, :3] = current_mat_global[3, :3]  # This seems like a hack, but gets rid of the translation component
            # selected_node.matrix = new_mat
            # selected_node.matrix_global = new_mat
            # # Transorm is in node matrix, matrix global, children -> patch's node transform, mesh transform
            # # There is a mesh in the child node for the patch and also a mesh at the entity level -> double version

            center, radius = self.selected_entity.node.mesh.bounding_sphere

            # Translate the pivot point to the center of the bounding sphere
            current_pivot = current_mat_local[3, :3]
            current_position = current_mat_global[3, :3]
            pivot_translation = np.identity(4)
            pivot_translation[3, :3] = -center
            new_mat = np.matmul(current_mat_global, pivot_translation)

            # Perform the rotation around the pivot point
            quat = self.get_world_rotation(x, y, x_last, y_last)
            rotation_matrix = np.array(quat.matrix44)
            # rotation_matrix = self.get_world_rotation(x, y, x_last, y_last)
            new_mat = np.matmul(new_mat, rotation_matrix)

            # Translate the object back to its original position
            inverse_pivot_translation = np.identity(4)
            inverse_pivot_translation[3, :3] = center
            new_mat = np.matmul(new_mat, inverse_pivot_translation)
            selected_node.matrix_global = new_mat
            # transformation = np.matmul(np.matmul(inverse_pivot_translation, rotation_matrix), pivot_translation)
            # selected_node.matrix_global = np.matmul(current_mat_global, transformation)

        else:
            dx, dy, dz = self.get_world_translations(x, y, x_last, y_last)
            translation_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, dz, 1]])
            selected_node.matrix_global = np.matmul(current_mat_global, translation_mat)

    def mouse_release_event(self, x: int, y: int, button: int):
        """On release, officially send request to move the object"""

        # Log for debugging events
        print(f"Mouse Release: {x}, {y}, {button}")

        # Pass event to gui
        self.gui.mouse_release_event(x, y, button)

        # Calculate vectors and move if applicable
        if self.selected_entity and self.last_click != (x, y):
            preview = self.selected_entity.node.matrix_global
            old = self.selected_entity.node.children[0].matrix_global

            try:
                # Rotation
                if not np.array_equal(preview[:3, :3], old[:3, :3]):
                    quat = Quaternion.from_matrix(preview)
                    self.selected_entity.set_rotation(quat.xyzw.tolist())

                # Position
                if not np.array_equal(preview[3, :3], old[3, :3]):
                    x, y, z = preview[3, :3].astype(float)
                    self.selected_entity.set_position([x, y, z])
                    self.update_widgets(old, preview)

            # Server doesn't support these injected methods
            except AttributeError as e:
                logging.warning(f"Dragging {self.selected_entity} failed: {e}")
                self.selected_entity.node.matrix = self.selected_entity.node.children[0].matrix
                self.selected_entity.node.matrix_global = old

            # Turn off ghosting effect
            self.selected_entity.node.mesh.ghosting = False

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

    def add_widgets(self):
        """Renders x, y, and z handle widgets for moving entities

        Idea: get bounding box and render widgets at the center of each positive face
              put the node as another child of the same selection node
        """

        center, radius = self.selected_entity.node.mesh.bounding_sphere

        widget_mesh = self.load_scene("arrow.obj").meshes[0]
        widget_mesh.mesh_program = programs.PhongProgram(self, 3)
        widget_mesh.name = "Widget Mesh"
        widget_mesh.norm_factor = (2 ** 32) - 1
        widget_mesh.entity_id = self.selected_entity.id
        vao = widget_mesh.vao

        # Add default colors
        default_colors = [1.0, 1.0, 1.0, 1.0] * vao.vertex_count
        buffer_data = np.array(default_colors, np.int8)
        vao.buffer(buffer_data, '4u1', 'in_color')

        # Add default textures
        default_texture_coords = [0.0, 0.0] * vao.vertex_count
        buffer_data = np.array(default_texture_coords, np.single)
        vao.buffer(buffer_data, '2f', 'in_texture')

        # Add instances, position, color, rotation quaternion, scale
        instances = [
            [
                [radius, 0, 0, 1],                              # Centered at edge of bbox
                [1.0, 0.0, 0.0, 0.5],                           # Red
                [0.7071, 0.7071, 0.0, 0],                       # Rotate 90 degrees around z
                [.1 * radius, .1 * radius, .1 * radius, 1]      # Scale proportionally
            ],
            [
                [0, radius, 0, 1],                              # Centered at edge of bbox
                [0.0, 0.0, 1.0, 0.5],                           # Blue
                [0.0, 0.0, 0.0, 1.0],                           # No rotation
                [.1 * radius, .1 * radius, .1 * radius, 1]      # Scale proportionally
            ],
            [
                [0, 0, radius, 1],                              # Centered at edge of bbox
                [0.0, 1.0, 0.0, 0.5],                           # Green
                [0.7071, 0.0, 0.0, -0.7071],                    # Rotate 90 degrees around x
                [.1 * radius, .1 * radius, .1 * radius, 1]      # Scale proportionally
            ]
        ]
        instance_data = np.array(instances, np.float32)
        vao.buffer(instance_data, '16f/i', 'instance_matrix')

        # Add widgets to scene, matrix moves to center of bounding sphere
        mat = np.identity(4, np.float32)
        mat[3, :3] = center
        widget_node = mglw.scene.Node("Widgets", mesh=widget_mesh, matrix=mat)
        widget_mesh.ghosting = False
        widget_mesh.has_bounding_sphere = False

        self.add_node(widget_node)

    def update_widgets(self, old_global, new_global):

        # Update bounding sphere in mesh
        selected_mesh = self.selected_entity.node.mesh
        old_center, old_radius = selected_mesh.bounding_sphere

        # Homogenize
        old_center = np.array([old_center[0], old_center[1], old_center[2], 1])

        # Transform center back to local space
        old_center_local = np.matmul(old_center, np.linalg.inv(old_global))

        # Transform back to world space with new transform
        new_center = np.matmul(old_center_local, new_global)
        new_center = new_center[:3]
        selected_mesh.bounding_sphere = (new_center, old_radius)

        # Update the widget transforms
        self.scene.find_node("Widgets").matrix[3, :3] = new_center
        self.update_matrices()

    def remove_widgets(self):

        # Remove widgets from scene
        widget_node = self.scene.find_node("Widgets")
        self.remove_node(widget_node)

    def render_scene_to_framebuffer(self, x, y):

        # Clear the framebuffer to max value 32 bit int
        self.fbo.clear()

        # Swap mesh programs to the frame select program
        old_programs = {}
        for mesh in self.scene.meshes:
            old_programs[(mesh.entity_id, mesh.name)] = mesh.mesh_program  # Save the old program
            mesh.mesh_program = programs.FrameSelectProgram(self, mesh.mesh_program.num_instances)

        self.fbo.use()
        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix
        )

        # Swap back to the old programs
        for mesh in self.scene.meshes:
            mesh.mesh_program = old_programs[(mesh.entity_id, mesh.name)]

        # Show pillow image for debugging
        # img = Image.frombytes('RGBA', (self.wnd.width, self.wnd.height), self.fbo.read(components=4))
        # img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # img.show()

        return self.fbo.read(components=4, viewport=(x, self.wnd.height-y, 1, 1), dtype='u4')

    def render(self, time: float, frametime: float):
        """Renders a frame to on the window
        
        Most work done in the draw function which draws each node in the scene.
        When drawing each node, the mesh is drawn, using the mesh program.
        At each frame, the callback_queue is checked so the client can update the render
        Note: each callback has the window as the first arg
        """

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)  # was raising problem in IDE but seemed to work

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
                clicked, self.draw_bs = imgui.checkbox("Show Bounding Boxes", self.draw_bs)

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

                # Lighting Settings
                imgui.menu_item("Lighting Settings", None, False, True)
                changed, self.default_lighting = imgui.checkbox("Default Lighting", self.default_lighting)

                # Widget Settings
                imgui.menu_item("Widget Settings", None, False, True)
                changed, self.move_widgets = imgui.checkbox("Movement Widgets", self.move_widgets)
                changed, self.rotate_widgets = imgui.checkbox("Rotation Widgets", self.rotate_widgets)
                changed, self.scale_widgets = imgui.checkbox("Scaling Widgets", self.scale_widgets)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        # Scene Info
        imgui.begin("Basic Info")
        imgui.text(f"Camera Position: {self.camera_position}")
        imgui.text(f"Press 'Space' to toggle camera/GUI")
        imgui.text(f"Click and drag an entity to move it")
        imgui.text(f"Hold 'r' while dragging to rotate an entity")
        _, self.draw_bs = imgui.checkbox("Show Bounding Boxes", self.draw_bs)
        imgui.end()

        # Render Document Methods and Signals or selection
        if self.selected_entity is None:
            self.render_document()
        else:
            imgui.begin("Selection")
            self.selected_entity.gui_rep()
            imgui.text(f"Instance: {self.selected_instance}")
            imgui.end()
