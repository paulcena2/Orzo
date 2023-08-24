"""Module Defining Default Delegates and Delegate Related Classes"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penne.core import Client

import io
import urllib.request
import json
import copy

from penne import *

import moderngl_window as mglw
import moderngl
import numpy as np
import quaternion
from PIL import Image as img
import imgui

from . import programs


@dataclass
class FormatInfo:
    """Information about a format

    Helpful for mapping between noodle and moderngl formats
    """
    num_components: int
    format: str
    size: int  # in bytes


@dataclass
class ChangeTracker:
    """Tracks changes to an entity's transform"""
    translation: bool
    rotation: bool
    scale: bool

    def __init__(self):
        self.translation = False
        self.rotation = False
        self.scale = False

    def reset(self):
        self.translation = False
        self.rotation = False
        self.scale = False


# (num components, format per component, size per component)
FORMAT_MAP = {
    "U8": FormatInfo(1, 'u1', 1),
    "U16": FormatInfo(1, 'u2', 2),
    "U32": FormatInfo(1, 'u4', 4),
    "U16VEC2": FormatInfo(2, 'u2', 2),
    "U8VEC4": FormatInfo(4, 'u1', 1),
    "VEC2": FormatInfo(2, 'f', 4),
    "VEC3": FormatInfo(3, 'f', 4),
    "VEC4": FormatInfo(4, 'f', 4)
}

NP_FORMAT_MAP = {
    "U8": np.int8,
    "U16": np.int16,
    "U32": np.int32,
    "U8VEC4": np.int8,
    "U16VEC2": np.int16,
    "VEC2": np.single,
    "VEC3": np.single,
    "VEC4": np.single,
    "MAT3": np.single,
    "MAT4": np.single
}

MODE_MAP = {
    "TRIANGLES": moderngl.TRIANGLES,
    "POINTS": moderngl.POINTS,
    "LINES": moderngl.LINES,
    "LINE_LOOP": moderngl.LINE_LOOP,
    "LINE_STRIP": moderngl.LINE_STRIP,
    "TRIANGLE_STRIP": moderngl.TRIANGLE_STRIP
}

# Editor hint -> gui component, parameters to gui component, default values for the input
HINT_MAP = {
    "noo::any": (imgui.core.input_text, ("Any", 256), ""),
    "noo::text": (imgui.core.input_text, ("Text", 256), ""),
    "noo::integer": (imgui.core.input_int, "Int", ""),
    "noo::real": (imgui.core.input_float, ["Real"], 1.0),
    "noo::array": (imgui.core.input_text, ["[Array]", 256], ["[]"]),
    "noo::map": (imgui.core.input_text, ["{dict}", "{}", 256]),
    "noo::any_id": (imgui.core.input_int2, ["Id"], [0, 0]),
    "noo::entity_id": (imgui.core.input_int2, ["Entity Id"], [0, 0]),
    "noo::table_id": (imgui.core.input_int2, ["Table Id"], [0, 0]),
    "noo::plot_id": (imgui.core.input_int2, ["Plot Id"], [0, 0]),
    "noo::method_id": (imgui.core.input_int2, ["Method Id"], [0, 0]),
    "noo::signal_id": (imgui.core.input_int2, ["Signal Id"], [0, 0]),
    "noo::image_id": (imgui.core.input_int2, ["Image Id"], [0, 0]),
    "noo::sampler_id": (imgui.core.input_int2, ["Sampler Id"], [0, 0]),
    "noo::texture_id": (imgui.core.input_int2, ["Texture Id"], [0, 0]),
    "noo::material_id": (imgui.core.input_int2, ["Material Id"], [0, 0]),
    "noo::light_id": (imgui.core.input_int2, ["Light Id"], [0, 0]),
    "noo::buffer_id": (imgui.core.input_int2, ["Buffer Id"], [0, 0]),
    "noo::bufferview_id": (imgui.core.input_int2, ["Buffer View Id"], [0, 0]),
    "noo::range(a,b,c)": (imgui.core.input_float3, ["Range (a->b) step by c", 0, 0, 0]),
}


class MethodDelegate(Method):
    """Delegate representing a method which can be invoked on the server

    Attributes:
        client (client object): 
            client delegate is a part of
    """
    current_args: Optional[dict] = {}

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.text(self.name)

    def invoke_rep(self, on_delegate=None):

        # Main Method Rep
        if imgui.button(f"Invoke {self.name}"):

            # Invoke method if no args, otherwise open popup for arg input
            if self.arg_doc:
                imgui.open_popup(f"Invoke {self.name}")
            else:
                self.client.invoke_method(self.name, [], context=get_context(on_delegate))

        # Add more description to main window
        imgui.core.push_text_wrap_pos()
        imgui.text(f"Docs: {self.doc}")
        imgui.core.pop_text_wrap_pos()
        imgui.separator()

        # Popup window
        if imgui.begin_popup(f"Invoke {self.name}"):

            imgui.text("Input Arguments")
            imgui.separator()
            for arg in self.arg_doc:
                imgui.text(arg.name.upper())

                # Get input block from state or get default vals from map
                component, parameters, vals = self.current_args.get(arg.name, (None, None, None))
                if not component:
                    try:
                        hint = arg.editor_hint if arg.editor_hint else "noo::any"
                        component, parameters, vals = HINT_MAP[hint]
                    except Exception:
                        raise Exception(f"Invalid Hint for {arg.name} arg")

                # Render GUI component from parameters and vals
                label, rest = parameters[0], parameters[1:]
                if isinstance(vals, list):
                    changed, values = component(label, *vals, *rest)
                else:
                    changed, values = component(f"{label} for {arg.name}", vals, *rest)

                if changed:
                    print(f"Changed {arg.name} to {values}")

                self.current_args[arg.name] = (component, parameters, values)
                imgui.text(arg.doc)
                imgui.separator()

            if imgui.button("Submit"):

                # Get vals and convert type if applicable
                final_vals = []
                for arg in self.current_args.values():
                    value = arg[2]
                    try:
                        clean_val = json.loads(value)
                    except Exception:
                        clean_val = value
                    final_vals.append(clean_val)

                context = get_context(on_delegate)
                logging.info(f"Invoking the method: {self.name} w/ args: {final_vals}")
                self.client.invoke_method(self.name, final_vals, context=context)
            imgui.end_popup()


class SignalDelegate(Signal):
    """Delegate representing a signal coming from the server

    Attributes:
        client (Client): 
            client delegate is a part of
    """

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.text(f"{self.name}")


class TableDelegate(Table):
    """Delegate representing a table

    Each table delegate corresponds with a table on the server
    To use the table, you must first subscribe 

    Attributes:
        client (Client): 
            weak ref to client to invoke methods and such
        selections (dict): 
            mapping of name to selection object
        signals (signals): 
            mapping of signal name to function
        name (str): 
            name of the table
        id (list): 
            id group for delegate in state and table on server
    """

    method_delegates: list = []
    signal_delegates: list = []

    def on_new(self, message: dict):
        self.method_delegates = [self.client.get_delegate(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_delegate(id) for id in self.signals_list]

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if not expanded:
            imgui.unindent()
            return

        imgui.text(f"Attached Methods: {self.methods_list}")
        if self.method_delegates:
            for method in self.method_delegates:
                method.invoke_rep(self)

        imgui.text(f"Attached Signals: {self.signals_list}")
        if self.signal_delegates:
            for signal in self.signal_delegates:
                signal.gui_rep()
        imgui.unindent()


class DocumentDelegate(Document):

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.text(f"{self.name}")
        imgui.text("Methods")
        for method in self.methods_list:
            self.client.get_delegate(method).gui_rep()


class EntityDelegate(Entity):
    """Delegate for overarching entities
    
    Can be container for storing meshes, lights, or plots

    In the scene, this is implemented with a moderngl-window node. This node has a local and global matrix which
    are updated to reflect the entities transform. The node also has a list of children which contain patches for
    the meshes. If the entity has a mesh, a version is stored in the entities node to use as a ghost preview. Another
    is stored in the child node. This may misbehave with multiple patches, but for now it works well enough.

    Lights are attached to the scene and updated with the entity's transform.

    !!! note

        All transforms should be column major order (OpenGL style). This is not the same as the
        numpy default row major, so be careful when using numpy arrays. Transposing then multiplying
        is the same as swapping the order, so many times you'll see the order swapped.
    
    Attributes:
        name (str): Name of the entity, defaults to 'No-Name Entity'
    """

    node: Optional[mglw.scene.Node] = None
    patch_nodes: list = []
    light_delegates: list[LightDelegate] = []
    geometry_delegate: Optional[GeometryDelegate] = None
    methods_list: Optional[List[MethodID]] = []
    signals_list: Optional[List[SignalID]] = []
    method_delegates: Optional[list[MethodDelegate]] = None
    signal_delegates: Optional[list[SignalDelegate]] = None
    table_delegate: Optional[TableDelegate] = None
    num_instances: Optional[int] = 0
    np_transform: Optional[np.ndarray] = np.eye(4)

    # These correspond to current state / preview
    translation: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0])
    rotation: Optional[np.quaternion] = np.quaternion(1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    scale: Optional[np.ndarray] = np.array([1.0, 1.0, 1.0])
    changed: Optional[ChangeTracker] = ChangeTracker()

    def render_entity(self, window):
        """Render the mesh associated with this delegate
        
        Will be called as callback from window. Largely hands the work off to the geometry delegate
        """

        assert self.render_rep is not None, "Entity must have a render representation to render"

        # Prepare Mesh
        geometry = self.client.get_delegate(self.render_rep.mesh)
        self.geometry_delegate = geometry
        self.patch_nodes, self.num_instances = geometry.render(window, self)

        # Add geometry patch nodes as children to main
        for node in self.patch_nodes:
            node.matrix = np.identity(4, np.float64)
            window.add_node(node, parent=self.node)

    def remove_from_render(self, window):
        """Remove mesh from render"""
        window.remove_node(self.node)

    def attach_lights(self, window):
        """Callback to handle lights attached to an entity

        Light info is stored directly in the window. This info is used in the
        mesh program to pass lights into the shader.
        """

        self.light_delegates = []  # Reset in case of update
        for light_id in self.lights:

            # Keep track of light delegates
            light_delegate = self.client.get_delegate(light_id)
            self.light_delegates.append(light_delegate)

            # Add positional and directional info to the light
            light_info = light_delegate.light_basics
            world_transform = self.get_world_transform()
            pos = np.matmul(np.array([0.0, 0.0, 0.0, 1.0]), world_transform)
            direction = np.matmul(np.array([0.0, 0.0, -1.0, 1.0]), world_transform)
            light_info["world_position"] = (pos[0] / pos[3], pos[1] / pos[3], pos[2] / pos[3])
            light_info["direction"] = (
                direction[0] / direction[3], direction[1] / direction[3], direction[2] / direction[3]
            )

            # Update State
            if light_id not in window.lights:
                window.lights[light_id] = light_info

    def update_lights(self, window):
        """Callback for updating lights on window and delegate"""
        self.remove_lights(window)
        self.attach_lights(window)

    def remove_lights(self, window):
        """Callback for removing lights from state"""

        for light_id in self.lights:
            del window.lights[light_id]

    def compose_transform(self, translation=None, rotation=None, scale=None):
        """Get a transform matrix given the current scale, rotation, and position

        Defaults to using current state, but can also accept args to use other values.
        Inputs should be numpy arrays or numpy-quaternions
        """
        if translation is None:
            translation = self.translation
        if rotation is None:
            rotation = self.rotation
        if scale is None:
            scale = self.scale

        transform = np.eye(4)
        transform[3, :3] = translation
        transform[:3, :3] = np.matmul(np.diag(scale), quaternion.as_rotation_matrix(rotation)).T
        return transform

    def decompose_transform(self):
        """Get the current scale, rotation, and position from the transform matrix"""
        self.translation = self.np_transform[3, :3]
        self.scale = np.linalg.norm(self.np_transform[:3, :3], axis=1)
        inverse_scale = np.linalg.inv(np.diag(self.scale))
        without_scale = np.matmul(inverse_scale, self.np_transform[:3, :3])
        self.rotation = quaternion.from_rotation_matrix(without_scale.T)

    def get_world_transform(self):
        """Get the current world transform for the entity"""
        return self.node.matrix_global

    def set_up_node(self, window):

        # Create node with local transform
        self.node = mglw.scene.Node(f"{self.id}'s Node", matrix=self.np_transform)

        # Update Scene / State
        if self.parent:
            window.add_node(self.node, parent=self.client.get_delegate(self.parent).node)
        else:
            window.add_node(self.node, parent=None)

    def update_matrices(self, window):
        """Update global matrices for all nodes in scene"""
        window.update_matrices()

    def on_new(self, message: dict):

        # Create node - even lights are represented with a node
        self.client.callback_queue.put((self.set_up_node, []))

        # Reformat transform and keep separate components
        if self.transform:
            # This keeps in col major order for MGLW
            self.np_transform = np.array(self.transform, np.float32).reshape(4, 4)

            # Decompose transform into components - transform, scales, unit quaternion
            self.decompose_transform()

        # Render mesh
        if self.render_rep:
            self.client.callback_queue.put((self.render_entity, []))

        # Attach lights to scene
        if self.lights:
            self.client.callback_queue.put((self.attach_lights, []))

        # Hook up methods and signals
        if self.methods_list:
            inject_methods(self, self.methods_list)
        if self.signals_list:
            inject_signals(self, self.signals_list)
        self.method_delegates = [self.client.get_delegate(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_delegate(id) for id in self.signals_list]

    def on_update(self, message: dict):

        # Recursively update mesh transforms if changed
        if "transform" in message or "parent" in message:
            self.np_transform = np.array(self.transform, np.float32).reshape(4, 4)
            self.node.matrix = self.np_transform
            self.client.callback_queue.put((self.update_matrices, []))

            self.decompose_transform()
            self.changed.reset()

            # Update light positions
            if self.lights:
                self.client.callback_queue.put((self.update_lights, []))

        # New mesh on entity
        if "render_rep" in message:
            self.client.callback_queue.put((self.remove_from_render, []))
            self.client.callback_queue.put((self.render_entity, []))

        # Update lights attached and light positions
        if "lights" in message:
            self.client.callback_queue.put((self.update_lights, []))

        # Update attached methods and signals from updated lists
        if "methods_list" in message:
            inject_methods(self, self.methods_list)
            self.method_delegates = [self.client.get_delegate(id) for id in self.methods_list]
        if "signals_list" in message:
            inject_signals(self, self.signals_list)
            self.signal_delegates = [self.client.get_delegate(id) for id in self.signals_list]

    def on_remove(self, message: dict):

        if self.render_rep:
            self.client.callback_queue.put((self.remove_from_render, []))

        if self.lights:
            self.client.callback_queue.put((self.remove_lights, []))

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if not expanded:
            imgui.unindent()
            return

        if self.geometry_delegate:
            self.geometry_delegate.gui_rep()
            if self.render_rep.instances:
                self.client.get_delegate(self.render_rep.instances.view).gui_rep()
            imgui.text(f"Num Instances: {self.num_instances}")
        if self.table_delegate:
            self.table_delegate.gui_rep()
        if self.light_delegates:
            for light in self.light_delegates:
                light.gui_rep()

        if self.transform is not None:
            imgui.text(f"Transform: {self.np_transform}")

        imgui.text(f"Attached Methods: {self.methods_list}")
        if self.method_delegates:
            for method in self.method_delegates:
                method.invoke_rep(self)

        imgui.text(f"Attached Signals: {self.signals_list}")
        if self.signal_delegates:
            for signal in self.signal_delegates:
                signal.gui_rep()
        imgui.unindent()


class PlotDelegate(Plot):

    method_delegates: list = []
    signal_delegates: list = []

    def on_new(self, message: dict):
        self.method_delegates = [self.client.get_delegate(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_delegate(id) for id in self.signals_list]

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if not expanded:
            imgui.unindent()
            return

        imgui.text(f"Attached Methods: {self.methods_list}")
        if self.method_delegates:
            for method in self.method_delegates:
                method.invoke_rep(self)

        imgui.text(f"Attached Signals: {self.signals_list}")
        if self.signal_delegates:
            for signal in self.signal_delegates:
                signal.gui_rep()
        imgui.unindent()


class GeometryDelegate(Geometry):

    @staticmethod
    def calculate_bounding_sphere(vertices, instance_positions, entity):
        """Calculate bounding sphere for a mesh

        If we are dealing with instance rendering, assume instances are small and box them all in.
        If dealing with vertices calculate it around the mesh. Center is more like a center of mass
        as opposed to a geometric center.
        """

        if instance_positions is not None:
            points = instance_positions
        else:
            points = vertices

        # Calculate the center of the bounding sphere
        center = np.mean(points, axis=0)

        # Calculate the maximum distance from the center to any vertex
        max_distance = np.max(np.linalg.norm(points - center, axis=1))

        # Translate center to world space
        world_transform = entity.get_world_transform()
        center = np.matmul(np.array([*center, 1.0]), world_transform)

        return center[:3], max_distance

    @staticmethod
    def calculate_normals(vertices, indices):
        """Calculate Normals for a Mesh that doesn't have them

        Modeled off of three.js approach here: https://github.com/mrdoob/three.js/blob/dev/src/core/BufferGeometry.js
        """
        normals = np.zeros(vertices.shape, dtype=np.single)
        for triangle in indices:
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]
            normal = np.cross(v2 - v1, v0 - v1)
            normals[triangle[0]] += normal
            normals[triangle[1]] += normal
            normals[triangle[2]] += normal

        # normalize normals
        return normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    @staticmethod
    def extract_bytes(raw_bytes, attr_offset, length, attr_stride, attr_format):
        if attr_stride == 0:
            attr_stride = attr_format.size * attr_format.num_components
        starts = range(attr_offset, attr_offset + length, attr_stride)
        byte_chunks = [memoryview(raw_bytes)[start:start + (attr_format.size * attr_format.num_components)] for
                       start in starts]  # Slicing with memory view is much, much faster
        return b''.join(byte_chunks)

    @staticmethod
    def reformat_color(raw_bytes, color_format):
        """Reformat all colors to consistent u8vec4's"""

        if color_format == "U8VEC4":
            return raw_bytes

        vals = np.frombuffer(raw_bytes, dtype=NP_FORMAT_MAP[color_format])
        max_val = np.finfo(np.single).max
        vals *= max_val  # not sure about this

        if color_format == "VEC3":
            # Pad to 4
            grouped = vals.reshape((-1, 3))
            col = np.array([1] * len(grouped))
            vals = np.append(grouped, col, axis=1)

        reformatted = vals.astype(np.int8).tobytes()
        return reformatted

    def set_up_indices(self, patch, vao):
        """Create index buffer for mesh

        Also returns the indices as a numpy array
        """
        index = patch.indices
        if index:
            index_view = self.client.get_delegate(index.view)
            index_format = FORMAT_MAP[index.format]
            index_size = index_format.size
            stride = index_view.stride if index.stride != 0 else index_size * index_format.num_components
            offset = index_view.offset + index.offset
            index_bytes = GeometryDelegate.extract_bytes(index_view.buffer_delegate.bytes, offset, index_view.length,
                                                         stride, index_format)
            indices = np.frombuffer(index_bytes, dtype=NP_FORMAT_MAP[index.format])
        else:
            # Non-indexed primitives just use range - 0, 1, 2, 3, etc...
            indices = np.arange(patch.vertex_count, dtype=np.single)
            index_bytes = indices.tobytes()
            index_size = 4  # four bytes / 32 bits for np.single

        # Create the buffer and store it in the VAO
        vao.index_buffer(index_bytes, index_size)
        return indices.reshape((-1, 3))

    def set_up_attributes(self, patch, mesh, indices):
        """Take care of setting up attributes for a mesh

        Breaks buffer into separate VAOs for each attribute
        """
        vertices = None

        # Break buffer up into VAO by attribute for robustness
        attribute_semantics = set([attr.semantic for attr in patch.attributes])
        for attribute in patch.attributes:
            view: BufferViewDelegate = self.client.get_delegate(attribute.view)
            buffer_bytes = view.buffer_delegate.bytes

            # Get format info
            format_info = FORMAT_MAP[attribute.format]
            buffer_format = f"{format_info.num_components}{format_info.format}"

            # Extract bytes and create buffer for attr
            attr_bytes = GeometryDelegate.extract_bytes(buffer_bytes, attribute.offset, view.length,
                                                        attribute.stride, format_info)

            # Reformat colors to consistent u8vec4's
            if attribute.semantic == "COLOR":
                attr_bytes = GeometryDelegate.reformat_color(attr_bytes, attribute.format)
                buffer_format = "4u1"

            # Calculate normals if they are missing
            if attribute.semantic == "POSITION":
                vertices = np.frombuffer(attr_bytes, np.float32).reshape((-1, 3))
                if "NORMAL" not in attribute_semantics:
                    normals = GeometryDelegate.calculate_normals(vertices, indices).flatten()
                    mesh.vao.buffer(normals, '3f', 'in_normal')
                    mesh.add_attribute("NORMAL", "in_normal", 3)

            attribute_name = f"in_{attribute.semantic.lower()}"  # What is passed into the shader
            mesh.vao.buffer(attr_bytes, buffer_format, attribute_name)
            mesh.add_attribute(attribute.semantic, attribute_name, format_info.num_components)
            # Add attribute doesn't let you specify type for some reason, i'm not sure if type is used somewhere
            # couldn't find it in docs / src. Another option would be to construct a dict and set mesh.attributes
            # manually.

        # Add default attributes for those that are missing
        if "COLOR" not in attribute_semantics:
            default_colors = [1.0, 1.0, 1.0, 1.0] * patch.vertex_count
            buffer_data = np.array(default_colors, np.int8)
            mesh.vao.buffer(buffer_data, '4u1', 'in_color')
            mesh.add_attribute("COLOR", "in_color", 4)  # Could check changing this to 1

        if "TEXTURE" not in attribute_semantics:
            default_texture_coords = [0.0, 0.0] * patch.vertex_count
            buffer_data = np.array(default_texture_coords, np.single)
            mesh.vao.buffer(buffer_data, '2f', 'in_texture')
            mesh.add_attribute("TEXTURE", "in_texture", 2)

        return vertices

    def set_up_instances(self, instances, mesh, window):
        """Set up instance buffer, program depends on whether there is instancing"""

        # Add instances to vao if applicable, also add appropriate mesh program
        if instances:
            instance_view = self.client.get_delegate(instances.view)
            instance_buffer = instance_view.buffer_delegate
            instance_bytes = instance_buffer.bytes
            mesh.vao.buffer(instance_bytes, '16f/i', 'instance_matrix')

            num_instances = int(instance_buffer.size / 64)  # 16 4 byte floats per instance
            mesh.mesh_program = programs.PhongProgram(window, num_instances)

            # Extract positions from instance buffer, used for calculating bounding sphere
            insts = np.frombuffer(instance_bytes, np.single).reshape((num_instances, 4, 4))
            positions = insts[:, 0, :3]

        else:
            positions = None
            num_instances = 0
            mesh.mesh_program = programs.PhongProgram(window, num_instances=-1)

        return positions, num_instances

    def render(self, window, entity):

        # Render each patch using the instances
        nodes = []
        num_instances = 0
        for patch in self.patches:
            node, num_instances = self.render_patch(patch, window, entity)
            nodes.append(node)
        return nodes, num_instances

    def render_patch(self, patch, window, entity):

        # Extract key attributes
        scene = window.scene
        transform = entity.np_transform
        instances = entity.render_rep.instances

        # Initialize VAO to store buffers and indices for this patch
        vao = mglw.opengl.vao.VAO(name=f"{self.name} Patch VAO", mode=MODE_MAP[patch.type])

        # Get Material - for now material delegate uses default texture
        material = self.client.get_delegate(patch.material)
        scene.materials.append(material.mglw_material)

        # Create the mesh object
        mesh = mglw.scene.Mesh(f"{self.name} Mesh", vao=vao, material=material.mglw_material)
        mesh.geometry_id = self.id
        mesh.entity_id = entity.id  # Can get delegate from mesh in click detection
        mesh.ghosting = False  # Ghosting turned off then will be turned on when dragged
        entity.node.mesh = mesh  # Add mesh to entity's node, used as preview and to delete from scene graph later

        # Create buffers for indices, attributes, and instances
        indices = self.set_up_indices(patch, mesh.vao)
        vertices = self.set_up_attributes(patch, mesh, indices)
        instance_positions, num_instances = self.set_up_instances(instances, mesh, window)

        # Calculate normals and bounding sphere if needed
        mesh.bounding_sphere = GeometryDelegate.calculate_bounding_sphere(vertices, instance_positions, entity)
        mesh.has_bounding_sphere = True

        # Add mesh as new node to scene graph
        mesh_copy = copy.copy(mesh)
        scene.meshes.append(mesh)
        new_mesh_node = mglw.scene.Node(f"{entity.name}'s Mesh Preview", mesh=mesh_copy, matrix=transform)

        return new_mesh_node, num_instances

    def patch_gui_rep(self, patch: GeometryPatch):
        """Rep for patches to be nested in GUI"""
        imgui.text("Attributes")
        for attribute in patch.attributes:
            imgui.indent()
            imgui.text(attribute.semantic)
            imgui.text(f"From buffer {attribute.view}")
            expanded, visible = imgui.collapsing_header(f"More Info for {attribute.semantic}", visible=True)
            if expanded:
                imgui.text(f"Channel: {attribute.channel}")
                imgui.text(f"Offset: {attribute.offset}")
                imgui.text(f"Stride: {attribute.stride}")
                imgui.text(f"Format: {attribute.format}")
                imgui.text(f"Min Value: {attribute.minimum_value}")
                imgui.text(f"Max Value: {attribute.maximum_value}")
                imgui.text(f"Normalized: {attribute.normalized}")
            imgui.unindent()

        imgui.separator()
        imgui.text("Index Info")
        index = patch.indices
        index_view = self.client.get_delegate(index.view)
        index_view.gui_rep()
        imgui.text(f"Count: {index.count}")
        imgui.text(f"Offset: {index.offset}")
        imgui.text(f"Stride: {index.stride}")
        imgui.text(f"Format: {index.format}")
        imgui.separator()

        if patch.material:
            self.client.get_delegate(patch.material).gui_rep()

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:

            imgui.indent()
            for patch in self.patches:
                self.patch_gui_rep(patch)
            imgui.unindent()

        imgui.unindent()


class LightDelegate(Light):
    """Delegate to store basic info associated with that light"""

    light_basics: dict = {}

    def on_new(self, message: dict):

        # Add info based on light type
        color = self.color
        if self.point:
            light_type = 0
            info = (self.intensity, self.point.range, 0.0, 0.0)
        elif self.spot:
            light_type = 1
            info = (self.intensity, self.spot.range, self.spot.inner_cone_angle_rad, self.spot.outer_cone_angle_rad)
        else:
            light_type = 2
            info = (self.intensity, self.directional.range, 0.0, 0.0)

        # Arrange info into dict to store
        self.light_basics = {
            "color": color.as_rgb_tuple(alpha=True),
            "ambient": (.1, .1, .1),
            "type": light_type,
            "info": info,
        }

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            for key, val in self.light_basics.items():
                imgui.text(f"{key.upper()}: {val}")
        imgui.unindent()


class MaterialDelegate(Material):
    """Delegate representing a Noodles Material"""

    texture_delegate: TextureDelegate = None
    color: tuple = None
    mglw_material: mglw.scene.Material = None

    def set_up_texture(self, window):
        """Set up texture for base color if applicable"""

        # Get texture
        self.texture_delegate = self.client.get_delegate(self.pbr_info.base_color_texture.texture)
        mglw_texture = self.texture_delegate.mglw_texture

        # Hook texture up to sampler
        mglw_sampler = self.texture_delegate.sampler_delegate.mglw_sampler
        mglw_sampler.texture = mglw_texture

        # Make sure wrapping flags match
        mglw_texture.repeat_x = mglw_sampler.repeat_x
        mglw_texture.repeat_y = mglw_sampler.repeat_y

        self.mglw_material.mat_texture = mglw.scene.MaterialTexture(mglw_texture, mglw_sampler)

    def on_new(self, message: dict):
        """"Create mglw_material from noodles message"""

        self.color = self.pbr_info.base_color.as_rgb_tuple(alpha=True)

        material = mglw.scene.Material(f"{self.name}")
        material.color = self.color

        # Only worrying about base_color_texture, need to delay in queue to allow for other setup - better solution?
        if self.pbr_info.base_color_texture:
            self.client.callback_queue.put((self.set_up_texture, []))

        material.double_sided = self.double_sided
        self.mglw_material = material

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            imgui.text(f"Color: {self.color}")
            self.texture_delegate.gui_rep() if self.texture_delegate else imgui.text(f"No Texture")
        imgui.unindent()


class ImageDelegate(Image):

    size: tuple = (0, 0)
    components: int = None
    bytes: bytes = None
    texture_id: int = None
    component_map: dict = {
        "RGB": 3,
        "RGBA": 4
    }

    def on_new(self, message: dict):

        # Get Bytes from either source present
        if self.buffer_source:
            buffer = self.client.get_delegate(self.buffer_source)
            self.bytes = buffer.bytes
        else:
            # beginning, end = self.uri_source.split("30043s")
            # self.uri_source = beginning + "30043s.local" + end
            with urllib.request.urlopen(self.uri_source) as response:
                self.bytes = response.read()

        im = img.open(io.BytesIO(self.bytes))
        # im = im.transpose(img.FLIP_TOP_BOTTOM)
        self.size = im.size
        self.components = self.component_map[im.mode]
        self.bytes = im.tobytes()

    def gui_rep(self):
        """Representation to be displayed in GUI"""

        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            imgui.image(self.texture_id, *self.size)
            imgui.text(f"Size: {self.size}")
            imgui.text(f"Components: {self.components}")
        imgui.unindent()


class TextureDelegate(Texture):

    image_delegate: ImageDelegate = None
    sampler_delegate: SamplerDelegate = None
    mglw_texture: moderngl.Texture = None

    def set_up_texture(self, window):
        image = self.client.get_delegate(self.image)
        self.image_delegate = image
        self.mglw_texture = window.ctx.texture(image.size, image.components, image.bytes)
        window.gui.register_texture(self.mglw_texture)
        self.image_delegate.texture_id = self.mglw_texture.glo
        # self.image_delegate.texture_id = 2

    def on_new(self, message: dict):

        self.client.callback_queue.put((self.set_up_texture, []))

        if self.sampler:
            self.sampler_delegate = self.client.get_delegate(self.sampler)

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            self.image_delegate.gui_rep()
            self.sampler_delegate.gui_rep() if self.sampler else imgui.text(f"No Sampler")
        imgui.unindent()


class SamplerDelegate(Sampler):

    rep_x: bool = None
    rep_y: bool = None
    mglw_sampler: moderngl.Sampler = None

    FILTER_MAP: dict = {
        "NEAREST": moderngl.NEAREST,
        "LINEAR": moderngl.LINEAR,
        "LINEAR_MIPMAP_LINEAR": moderngl.LINEAR_MIPMAP_LINEAR,
    }

    SAMPLER_MODE_MAP: dict = {
        "CLAMP_TO_EDGE": False,
        "REPEAT": True,
        "MIRRORED_REPEAT": True  # This is off but mglw only allows for boolean
    }

    def set_up_sampler(self, window):

        self.rep_x = self.SAMPLER_MODE_MAP[self.wrap_s]
        self.rep_y = self.SAMPLER_MODE_MAP[self.wrap_t]

        self.mglw_sampler = window.ctx.sampler(
            filter=(self.FILTER_MAP[self.min_filter], self.FILTER_MAP[self.mag_filter]),
            repeat_x=self.rep_x,
            repeat_y=self.rep_y,
            repeat_z=False
        )

    def on_new(self, message: dict):
        self.client.callback_queue.put((self.set_up_sampler, []))

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            imgui.text(f"Min Filter: {self.min_filter}")
            imgui.text(f"Mag Filter: {self.mag_filter}")
            imgui.text(f"Repeat X: {self.rep_x}")
            imgui.text(f"Repeat Y: {self.rep_y}")
        imgui.unindent()


class BufferDelegate(Buffer):
    """Stores Buffer Info for Easier Access"""

    bytes: bytes = None

    def on_new(self, message: dict):

        if self.inline_bytes:
            self.bytes = self.inline_bytes
        elif self.uri_bytes:
            # beginning, end = self.uri_bytes.split("30043s")
            # self.uri_bytes = beginning + "30043s.local" + end
            with urllib.request.urlopen(self.uri_bytes) as response:
                self.bytes = response.read()
        else:
            raise Exception("Malformed Buffer Message")

    def gui_rep(self):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{self.name} {self.id.slot, self.id.gen}", visible=True)
        if expanded:
            imgui.text(f"Size: {self.size} bytes")
            imgui.text(f"Bytes: {self.bytes[:4]}...{self.bytes[-4:]}")
        imgui.unindent()


class BufferViewDelegate(BufferView):
    """Stores pointer to buffer for easier access"""

    buffer_delegate: BufferDelegate = None

    def on_new(self, message: dict):
        self.buffer_delegate: BufferDelegate = self.client.get_delegate(self.source_buffer)

    def gui_rep(self, description=""):
        """Representation to be displayed in GUI"""
        imgui.indent()
        expanded, visible = imgui.collapsing_header(f"{description}{self.name} {self.id.slot, self.id.gen}",
                                                    visible=True)
        if expanded:
            self.buffer_delegate.gui_rep()
            imgui.text(f"Type: {self.type}")
            imgui.text(f"Offset: {self.offset}")
            imgui.text(f"Length: {self.length}")
        imgui.unindent()


delegate_map = {
    Entity: EntityDelegate,
    Table: TableDelegate,
    Plot: PlotDelegate,
    Signal: SignalDelegate,
    Method: MethodDelegate,
    Material: MaterialDelegate,
    Geometry: GeometryDelegate,
    Light: LightDelegate,
    Image: ImageDelegate,
    Texture: TextureDelegate,
    Sampler: SamplerDelegate,
    Buffer: BufferDelegate,
    BufferView: BufferViewDelegate,
    Document: DocumentDelegate
}
