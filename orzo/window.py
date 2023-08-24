import logging
import os
from time import time

import moderngl_window as mglw
import moderngl
import numpy as np
import quaternion
from pathlib import Path
import imgui
from imgui.integrations.pyglet import create_renderer
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import penne

from orzo import programs
from orzo.delegates import delegate_map


current_dir = os.path.dirname(__file__)

# Rendering radius
SKYBOX_RADIUS = 1000.0

# Shader
DEFAULT_SHININESS = 10.0
DEFAULT_SPEC_STRENGTH = 0.2

# Widgets
WIDGET_SCALE = [0.1, 0.1, 0.1, 1.0]
X_WIDGET_COLOR = [1.0, 0.0, 0.0, 1.0]
Y_WIDGET_COLOR = [0.0, 1.0, 0.0, 1.0]
Z_WIDGET_COLOR = [0.0, 0.0, 1.0, 1.0]
X_WIDGET_ROTATION = [0.7071, 0.7071, 0.0, 0]
Y_WIDGET_ROTATION = [0.0, 0.0, 0.0, 1.0]
Z_WIDGET_ROTATION = [0.7071, 0.0, 0.0, -0.7071]

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up garbage collection
        self.ctx.gc_mode = 'context_gc'

        # Set Up Camera
        self.camera = mglw.scene.camera.KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.projection.update(near=0.1, far=1000.0)  # Range where camera will cutoff
        self.camera.mouse_sensitivity = 0.1
        self.camera.velocity = 3.0
        self.camera.zoom = 2.5
        self.camera_position = [0.0, 0.0, 0.0]

        # Set up Framebuffer - used for selection
        self.framebuffer = self.ctx.simple_framebuffer((self.wnd.width, self.wnd.height), dtype='u4')

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

        # Tried using imgui pyglet integration before, but switched to moderngl_window's integration
        # Could be worth taking another look at if event input problems are persistent
        # self.gui = create_renderer(self.wnd._window)

        # Set up GUI
        imgui.create_context()
        self.gui = ModernglWindowRenderer(self.wnd)
        self.address = "ws://localhost:50000"
        self.client_needs_shutdown = False
        self.args = {}
        self.selected_entity = None  # Current entity that is selected
        self.selected_instance = None  # Number instance that is selected
        self.rotating = False  # Flag for rotating entity on drag
        self.active_widget = None  # String for active widget type
        self.widgets_align_local = 0  # Whether widgets should be axis aligned or local
        self.translate_widgets = True  # Flags for enabling and disabling widgets
        self.rotate_widgets = True
        self.scale_widgets = True
        self.origin_centered = 0  # Where transforms like rotations should be centered

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

        # Get axis of rotation and angle
        axis = np.cross(click_vec, release_vec)
        axis /= np.sqrt(np.dot(axis, axis))
        angle = .05

        # Construct quaternion from axis vector, length is proportional to angle in radians
        return quaternion.from_rotation_vector(axis * angle)

    def key_event(self, key, action, modifiers):

        # Log for debugging events
        print(f"Key Entered: {key}, {action}, {modifiers}")

        # Pass event to gui
        self.gui.key_event(key, action, modifiers)

        # Move camera if enabled
        keys = self.wnd.keys
        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        # Handle key presses like quit and toggle camera
        if action == keys.ACTION_PRESS:
            if key == keys.C or key == keys.SPACE:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.P:
                self.timer.toggle_pause()

        # Rotation modifier
        if key == keys.R:
            if action == keys.ACTION_PRESS:
                self.rotating = True
            elif action == keys.ACTION_RELEASE:
                self.rotating = False

    def mouse_position_event(self, x: int, y: int, dx, dy):

        # Pass event to gui
        self.gui.mouse_position_event(x, y, dx, dy)

        # Move camera if enabled
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def mouse_press_event(self, x: int, y: int, button: int):

        # Add some timing info for debugging
        start_time = time()

        # Pass event to gui
        print("Click Registered")
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
            if self.selected_entity is not None:
                self.active_widget = None
                self.remove_widgets()
            self.selected_entity = None
            self.selected_instance = None
            end_time = time()
            print(f"Time to click nothing: {end_time - start_time}")
            return

        # Get widget type from hit
        if hit == 2:
            self.active_widget = "translation"
        elif hit == 3:
            self.active_widget = "rotation"
        elif hit == 4:
            self.active_widget = "scaling"
        else:
            self.active_widget = None

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

        end_time = time()
        print(f"Time to select: {end_time - start_time}")
        print(f"Active Widget: {self.active_widget}")

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Change appearance by changing the mesh's transform

        Essentially, the drag is changing the global matrix of the selected entity's node.
        This global matrix isn't really correct, given the local transforms, but it allows the
        preview to render temporarily.
        """

        # Add some timing info for debugging
        start_time = time()

        # Pass event to gui
        self.gui.mouse_drag_event(x, y, dx, dy)

        if not self.selected_entity or imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW) or (dx == 0 and dy == 0):
            return

        x_last, y_last = x - dx, y - dy
        entity = self.selected_entity

        # Turn on ghosting effect
        entity.node.mesh.ghosting = True

        # If widgeting, move using the widget's rules
        if self.active_widget is not None:
            dx, dy, dz = self.get_world_translations(x, y, x_last, y_last)
            self.handle_widget_movement(dx, dy, dz)

        # If r is held, rotate, if not translate
        elif self.rotating:
            center, radius = self.selected_entity.node.mesh.bounding_sphere

            rotation_quat = self.get_world_rotation(x, y, x_last, y_last)

            # Add translation to keep things centered
            shifted_origin = entity.translation - center
            inverse_rotation = np.quaternion(rotation_quat.w, -rotation_quat.x, -rotation_quat.y, -rotation_quat.z)
            rotated = quaternion.rotate_vectors(inverse_rotation, shifted_origin)
            entity.translation = rotated + center
            entity.changed.translation = True

            # Add the rotation to the current matrix
            entity.rotation = entity.rotation * rotation_quat
            entity.changed.rotation = True
            entity.node.matrix_global = entity.compose_transform()

        else:
            dx, dy, dz = self.get_world_translations(x, y, x_last, y_last)
            entity.translation += np.array([dx, dy, dz])
            entity.changed.translation = True
            entity.node.matrix_global = entity.compose_transform()

        end_time = time()
        print(f"Time to drag: {end_time - start_time}")

    def mouse_release_event(self, x: int, y: int, button: int):
        """On release, officially send request to move the object"""

        # Add some timing info for debugging
        start_time = time()

        # Pass event to gui
        print("Click Release Registered")
        self.gui.mouse_release_event(x, y, button)

        # If nothing is selected move on
        if self.selected_entity is None:
            return

        # Calculate vectors and move if applicable
        entity = self.selected_entity
        preview = entity.node.matrix_global
        old = entity.node.children[0].matrix_global
        if not np.array_equal(preview, old):

            try:

                if entity.changed.translation:
                    entity.set_position(entity.translation.tolist())

                if entity.changed.rotation:
                    # Revisit - Platter has negative w component
                    rearranged = [entity.rotation.x, entity.rotation.y, entity.rotation.z, entity.rotation.w]
                    entity.set_rotation(rearranged)

                if entity.changed.scale:
                    entity.set_scale(entity.scale.tolist())

                # Whether widgets should be updated
                plain_translation = not entity.changed.rotation and not entity.changed.scale
                rotate_around_origin = entity.changed.rotation and self.origin_centered
                local_rotation = entity.changed.rotation and self.widgets_align_local
                local_scale = entity.changed.scale and self.origin_centered
                if plain_translation or rotate_around_origin or local_rotation or local_scale:
                    self.update_widgets(old, preview)

                entity.changed.reset()

            # Server doesn't support these injected methods
            except AttributeError as e:
                logging.warning(f"Dragging {self.selected_entity} failed: {e}")
                self.selected_entity.node.matrix = self.selected_entity.node.children[0].matrix
                self.selected_entity.node.matrix_global = old

            # Turn off ghosting effect
            self.selected_entity.node.mesh.ghosting = False

        end_time = time()
        print(f"Time to release: {end_time - start_time}")

    def close(self):
        if self.client_needs_shutdown:
            self.client.shutdown()

    def resize(self, width: int, height: int):
        self.gui.resize(width, height)
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def unicode_char_entered(self, char):

        # Pass event to gui
        self.gui.unicode_char_entered(char)

    def create_widget_node(self, mesh, radius, offset):

        name = mesh[:-4]
        widget_mesh = self.load_scene(mesh).meshes[0]
        widget_mesh.mesh_program = programs.PhongProgram(self, 3)
        widget_mesh.name = f"noo::widget_{name}"
        widget_mesh.norm_factor = (2 ** 32) - 1
        widget_mesh.entity_id = self.selected_entity.id
        widget_mesh.ghosting = False
        widget_mesh.has_bounding_sphere = False
        vao = widget_mesh.vao

        # Add default colors
        default_colors = [1.0, 1.0, 1.0, 1.0] * vao.vertex_count
        buffer_data = np.array(default_colors, np.int8)
        vao.buffer(buffer_data, '4u1', 'in_color')

        # Add default textures
        default_texture_coords = [0.0, 0.0] * vao.vertex_count
        buffer_data = np.array(default_texture_coords, np.single)
        vao.buffer(buffer_data, '2f', 'in_texture')

        # Add instances -> position, color, rotation, scale
        # Instance trio is centered at origin and spaced out by radius
        instances = np.array([[[radius + offset, 0, 0, 1], X_WIDGET_COLOR, X_WIDGET_ROTATION, WIDGET_SCALE],
                              [[0, radius + offset, 0, 1], Y_WIDGET_COLOR, Y_WIDGET_ROTATION, WIDGET_SCALE],
                              [[0, 0, radius + offset, 1], Z_WIDGET_COLOR, Z_WIDGET_ROTATION, WIDGET_SCALE]],
                             np.float32)
        instances[:, 3, :3] *= .5 * radius  # Scale widget by radius
        vao.buffer(instances, '16f/i', 'instance_matrix')

        # Create the node
        widget_node = mglw.scene.Node(f"{name} widget", mesh=widget_mesh, matrix=np.identity(4))
        return widget_node

    def add_widgets(self):
        """Renders x, y, and z handle widgets for moving entities

        Create nodes for translation, rotation, and scaling widgets where each has 3 instances
        Then a parent node will store all the widgets. This parent node stores the translation
        that essentially does all the work to get things to line up with the entity
        """

        # Get entity info
        entity = self.selected_entity
        center, radius = entity.node.mesh.bounding_sphere

        # Create the parent node
        if self.widgets_align_local:
            mat = entity.compose_transform(scale=np.array([1.0, 1.0, 1.0]))  # Get global without scaling
        else:
            mat = np.identity(4, np.float32)
        mat[3, :3] = center
        widget_node = mglw.scene.Node("Widgets", matrix=mat)
        self.add_node(widget_node)

        # Create mesh nodes
        meshes, offsets = [], []
        if self.translate_widgets:
            meshes.append("cone.obj")
            offsets.append(.25 * radius)
        if self.rotate_widgets:
            meshes.append("torus.obj")
            offsets.append(0)
        if self.scale_widgets:
            meshes.append("tab.obj")
            offsets.append(0)

        for mesh, offset in zip(meshes, offsets):
            node = self.create_widget_node(mesh, radius, offset)
            self.add_node(node, parent=widget_node)

    def update_widgets(self, old_global, new_global):
        """Update widgets when the entity is moved

        Depending on the widget mode, the widgets will be moved in different ways
        In global mode, the widgets will always remain axis aligned, but in local mode
        the widgets will rotate with the entity.
        """

        # Update bounding sphere in mesh
        entity = self.selected_entity
        selected_mesh = entity.node.mesh
        old_center, old_radius = selected_mesh.bounding_sphere

        # Homogenize
        old_center = np.array([old_center[0], old_center[1], old_center[2], 1])

        # Transform center back to local space
        old_center_local = np.matmul(old_center, np.linalg.inv(old_global))

        # Transform back to world space with new transform
        new_center = np.matmul(old_center_local, new_global)
        new_center = new_center[:3]
        selected_mesh.bounding_sphere = (new_center, old_radius)
        for child in entity.node.children:
            child.mesh.bounding_sphere = (new_center, old_radius)

        # Update the widget transforms
        if self.widgets_align_local:
            new_mat = entity.compose_transform(scale=np.array([1.0, 1.0, 1.0]))  # Get global without scaling
            new_mat[3, :3] = new_center
            self.scene.find_node("Widgets").matrix = new_mat
        else:
            new_mat = np.identity(4)
            new_mat[3, :3] = new_center
            self.scene.find_node("Widgets").matrix = new_mat
        self.update_matrices()

    def remove_widgets(self):

        # Remove widgets from scene
        widget_node = self.scene.find_node("Widgets")
        self.remove_node(widget_node)

    def handle_widget_movement(self, dx, dy, dz):

        # Essentially a switch to guide drag to the proper handling
        widget_type = self.active_widget
        if widget_type == "translation":
            self.handle_widget_translation(dx, dy, dz)
        elif widget_type == "rotation":
            self.handle_widget_rotation(dx, dy, dz)
        elif widget_type == "scaling":
            self.handle_widget_scaling(dx, dy, dz)
        else:
            return

    def handle_widget_translation(self, dx, dy, dz):

        entity = self.selected_entity
        direction = self.selected_instance
        deltas = np.array([dx, dy, dz])

        # Translate in direction of widget
        if not self.widgets_align_local:
            entity.translation[direction] += deltas[direction]
        else:
            widget_direction = entity.node.matrix_global[direction, :3]
            magnitude = np.dot(deltas, widget_direction)
            entity.translation += widget_direction * magnitude

        # Add the translation to the current matrix
        entity.changed.translation = True
        entity.node.matrix_global = entity.compose_transform()

    def handle_widget_rotation(self, dx, dy, dz):
        """Rotate along the specified widget axis"""
        entity = self.selected_entity
        direction = self.selected_instance
        center, radius = entity.node.mesh.bounding_sphere
        widgets = self.scene.find_node("Widgets")

        # Get the rotation quaternion
        widget_vec = widgets.children[direction].matrix_global[direction, :3]
        vec = np.dot(widget_vec, np.array([dx, dy, dz])) * widget_vec
        rotation_quat = quaternion.from_rotation_vector(vec)

        # Apply Rotation
        entity.rotation = entity.rotation * rotation_quat
        entity.changed.rotation = True

        if not self.origin_centered:
            # Rotate around the center of the mesh - to pivot, then rotate, then go back to see new origin position
            to_pivot, from_pivot, rotation_mat = np.identity(4), np.identity(4), np.identity(4)
            to_pivot[3, :3] = -center
            from_pivot[3, :3] = center
            rotation_mat[:3, :3] = quaternion.as_rotation_matrix(rotation_quat).T
            rotation_mat = np.matmul(to_pivot, np.matmul(rotation_mat, from_pivot))
            entity.translation = np.matmul(np.array([entity.translation[0], entity.translation[1], entity.translation[2], 1]), rotation_mat)[:3]
            entity.changed.translation = True

        entity.node.matrix_global = entity.compose_transform()

    def handle_widget_scaling(self, dx, dy, dz):
        """Scale along the specified widget axis"""
        entity = self.selected_entity
        current_mat_global = entity.node.matrix_global
        direction = self.selected_instance
        center, radius = entity.node.mesh.bounding_sphere
        deltas = np.array([dx, dy, dz])
        origin = entity.node.matrix[3, :3]
        # origin = current_mat_global[3, :3]
        widgets = self.scene.find_node("Widgets")

        # Scale in direction of widget
        widget_vec = widgets.children[direction].matrix_global[direction, :3]
        magnitude = np.dot(deltas, widget_vec)
        rotated_scale = quaternion.rotate_vectors(entity.rotation, widget_vec)
        entity.scale += rotated_scale * magnitude

        # 1.
        # to_pivot, from_pivot, transform_mat = np.identity(4), np.identity(4), np.identity(4)
        # to_pivot[3, :3] = -center
        # from_pivot[3, :3] = center
        #
        # # Use pieces to construct the marginal transform that is being applied
        # rotated_scale *= (radius / 2)
        # marginal_scale = np.diag(rotated_scale * magnitude)
        # marginal_scale[0, 0] += 1
        # marginal_scale[1, 1] += 1
        # marginal_scale[2, 2] += 1
        # transform_mat[:3, :3] = marginal_scale
        # transform_mat = np.matmul(to_pivot, np.matmul(transform_mat, from_pivot))
        #
        # # Apply transform to the old translation
        # entity.translation = np.matmul(
        #     np.array([entity.translation[0], entity.translation[1], entity.translation[2], 1]), transform_mat)[:3]
        #entity.translation = np.matmul(np.array([origin[0], origin[1], origin[2], 1]), transform_mat)[:3]

        # 2.  problem is this origin also has some scaling baked in
        # old_scale = entity.scale - (rotated_scale * magnitude)
        # shifted_origin = (origin) - center
        # scaled = shifted_origin * old_scale
        # new_scaled = shifted_origin * entity.scale
        # entity.translation = entity.translation - (scaled - new_scaled)

        # If centered around center, then add translate to keep centered
        # Problem is when the origin placement is different
        # - for example if origin is already at center, then no need to translate
        if not self.origin_centered:
            entity.translation -= widget_vec * magnitude
            entity.changed.translation = True

        # Add the scaling to the current matrix
        entity.changed.scale = True
        entity.node.matrix_global = entity.compose_transform()

    def render_scene_to_framebuffer(self, x, y):

        # Swap mesh programs to the frame select program
        old_programs = {}
        for mesh in self.scene.meshes:
            old_programs[(mesh.entity_id, mesh.name)] = mesh.mesh_program  # Save the old program
            mesh.mesh_program = programs.FrameSelectProgram(self, mesh.mesh_program.num_instances)

        self.framebuffer.use()
        self.framebuffer.clear()
        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix
        )

        # Swap back to the old programs
        for mesh in self.scene.meshes:
            mesh.mesh_program = old_programs[(mesh.entity_id, mesh.name)]

        # Bind the default framebuffer to switch back
        self.ctx.screen.use()

        return self.framebuffer.read(components=4, viewport=(x, self.wnd.height-y, 1, 1), dtype='u4')

    def render(self, time: float, frametime: float):
        """Renders a frame to on the window
        
        Most work done in the draw function which draws each node in the scene.
        When drawing each node, the mesh is drawn, using the mesh program.
        At each frame, the callback_queue is checked so the client can update the render
        Note: each callback has the window as the first arg
        """
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

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

        # Show log in window if client is still not connected
        if self.client is None:
            self.render_login()
            imgui.render()
            self.gui.render(imgui.get_draw_data())
            return

        # Render GUI elements
        self.update_gui()
        imgui.render()
        self.gui.render(imgui.get_draw_data())

        while not self.client.callback_queue.empty():
            callback_info = self.client.callback_queue.get()
            callback, args = callback_info
            logging.info(f"Callback in render: {callback.__name__} \n\tw/ args: {args}")
            callback(self, *args)

        # Clean up any dead objects
        self.ctx.gc()

    def render_login(self):
        imgui.new_frame()
        imgui.begin("Connect to Server")
        imgui.text("Enter Websocket Address")
        changed, self.address = imgui.input_text("Address", self.address, 256)
        if imgui.button("Connect"):
            self.client = penne.Client(self.address, delegate_map)
            self.client.thread.start()  # Starts websocket connection in new thread
            self.client.connection_established.wait()
            self.client_needs_shutdown = True
        imgui.end()

        # Scene Info
        self.render_scene_info()

    def render_scene_info(self):
        # Scene Info
        imgui.begin("Basic Info")
        imgui.text(f"Camera Position: {self.camera_position}")
        imgui.text(f"Press 'Space' to toggle camera/GUI")
        imgui.text(f"Click and drag an entity to move it")
        imgui.text(f"Hold 'r' while dragging to rotate an entity")
        _, self.draw_bs = imgui.checkbox("Show Bounding Spheres", self.draw_bs)
        imgui.end()

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
                    self.close()
                imgui.end_menu()
            if imgui.begin_menu("Settings", True):

                # Bboxes
                clicked, self.draw_bs = imgui.checkbox("Show Bounding Spheres", self.draw_bs)

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

                # Toggle which are active
                changed_t, self.translate_widgets = imgui.checkbox("Movement Widgets", self.translate_widgets)
                changed_r, self.rotate_widgets = imgui.checkbox("Rotation Widgets", self.rotate_widgets)
                changed_s, self.scale_widgets = imgui.checkbox("Scaling Widgets", self.scale_widgets)
                if self.active_widget is not None and (changed_t or changed_r or changed_s):
                    self.remove_widgets()
                    self.add_widgets()

                # Toggle alignment
                clicked, self.widgets_align_local = imgui.combo(
                    "Widget Alignment", self.widgets_align_local, ["Global", "Local"]
                )
                if clicked and self.selected_entity:
                    preview = self.selected_entity.node.matrix_global
                    old = self.selected_entity.node.children[0].matrix_global
                    self.update_widgets(old, preview)

                # Toggle origin
                clicked, self.origin_centered = imgui.combo(
                    "Transformation Origin", self.origin_centered, ["Center of Mesh", "Object Origin"]
                )

                imgui.end_menu()
            imgui.end_main_menu_bar()

        self.render_scene_info()

        # Render Document Methods and Signals or selection
        if self.selected_entity is None:
            self.render_document()
        else:
            imgui.begin("Selection")
            self.selected_entity.gui_rep()
            imgui.text(f"Instance: {self.selected_instance}")
            imgui.end()
