"""Module Defining Default Delegates and Delegate Related Classes"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penne.core import Client

import io
import urllib.request
import json

from . import programs

from penne import *

import moderngl_window as mglw
import moderngl
import numpy as np
from PIL import Image as img
import imgui


@dataclass
class FormatInfo:
    num_components: int
    format: str
    size: int  # in bytes


FORMAT_MAP = {
    # (num components, format per component, size per component)
    "U8": FormatInfo(1, 'u1', 1),
    "U16": FormatInfo(1, 'u2', 2),
    "U32": FormatInfo(1, 'u4', 4),
    "U16VEC2": FormatInfo(2, 'u2', 2),
    "U8VEC4": FormatInfo(4, 'u1', 1),
    "VEC2": FormatInfo(2, 'f', 4),
    "VEC3": FormatInfo(3, 'f', 4),
    "VEC4": FormatInfo(4, 'f', 4)
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
    "noo::real": (imgui.core.input_float, "Real", 1.0),
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
    current_args = {}

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
                self.client.invoke_method(self.name, [])

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
                print(f"Invoking the method: {self.name} w/ args: {final_vals}")
                self.client.invoke_method(self.name, final_vals, context=context)
            imgui.end_popup()


class SignalDelegate(Signal):
    """Delegate representing a signal coming from the server

    Attributes:
        client (Client): 
            client delegate is a part of
    """

    def on_new(self, message: dict):
        pass

    def on_remove(self, message: dict):
        pass

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

    method_delegates = []
    signal_delegates = []

    def on_new(self, message: dict):
        self.method_delegates = [self.client.get_component(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_component(id) for id in self.signals_list]

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
            self.client.get_component(method).gui_rep()


class EntityDelegate(Entity):
    """Delegate for overarching entities
    
    Can be container for storing meshes, lights, or plots
    
    Attributes:
        name (str): Name of the entity, defaults to 'No-Name Entity'
    """

    nodes = []
    light_delegates: list[LightDelegate] = []
    geometry_delegate: GeometryDelegate = None
    methods_list: Optional[List[MethodID]] = []
    signals_list: Optional[List[SignalID]] = []
    method_delegates: list[MethodDelegate] = None
    signal_delegates: list[SignalDelegate] = None
    table_delegate: TableDelegate = None
    num_instances: int = 0

    def render_entity(self, window):
        """Render the mesh associated with this delegate
        
        Will be called as callback from window
        """

        # Prepare Mesh
        geometry = self.client.get_component(self.render_rep.mesh)
        self.geometry_delegate = geometry
        instances = self.render_rep.instances
        geometry.render(instances, window)

    def attach_lights(self, window):
        """Callback to handle lights attached to an entity"""
        for light_id in self.lights:

            # Keep track of light delegates
            light_delegate = self.client.get_component(light_id)
            self.light_delegates.append(light_delegate)

            # Add positional and directional info to the light
            light_info = light_delegate.light_basics
            world_transform = self.get_world_transform()
            pos = np.matmul(world_transform, np.array([0.0, 0.0, 0.0, 1.0]))
            direction = np.matmul(world_transform, np.array([0.0, 0.0, -1.0, 1.0]))
            light_info["world_position"] = (pos[0] / pos[3], pos[1] / pos[3], pos[2] / pos[3])
            light_info["direction"] = (
                direction[0] / direction[3], direction[1] / direction[3], direction[2] / direction[3]
            )

            # Update State
            if light_id not in window.lights:
                window.lights[light_id] = light_info

    def remove_lights(self, window):
        """Callback for removing lights from state"""

        for light_id in self.lights:
            del window.lights[light_id]

    def get_world_transform(self):
        """Recursive function to get world transform for an entity"""

        # Swap axis to go from col major -> row major order
        local_transform = np.array(self.transform).reshape(4, 4).swapaxes(0, 1)

        if self.parent:
            parent = self.client.get_component(self.parent)
            return np.matmul(parent.get_world_transform(), local_transform)
        else:
            return local_transform

    def remove_from_render(self, window):
        """Remove mesh from render"""

        # Need to test, enough to remove from render?
        scene = window.scene
        for node in self.nodes:
            scene.root_nodes[0].children.remove(node)
            scene.nodes.remove(node)
            scene.meshes.remove(node.mesh)

    def on_new(self, message: dict):

        if self.render_rep:
            self.client.callback_queue.put((self.render_entity, []))

        if self.lights:
            self.client.callback_queue.put((self.attach_lights, []))

        self.method_delegates = [self.client.get_component(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_component(id) for id in self.signals_list]

    def on_update(self, message: dict):

        if self.render_rep:
            self.client.callback_queue.put((self.remove_from_render, []))
            self.client.callback_queue.put((self.render_entity, []))

        if self.lights:
            self.client.callback_queue.put((self.attach_lights, []))
            self.client.callback_queue.put((self.remove_lights, []))

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
                self.client.get_component(self.render_rep.instances.view).gui_rep()
            imgui.text(f"Num Instances: {self.num_instances}")
        if self.table_delegate:
            self.table_delegate.gui_rep()
        if self.light_delegates:
            for light in self.light_delegates:
                light.gui_rep()

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

    method_delegates = []
    signal_delegates = []

    def on_new(self, message: dict):
        self.method_delegates = [self.client.get_component(id) for id in self.methods_list]
        self.signal_delegates = [self.client.get_component(id) for id in self.signals_list]

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
    def reformat_attr(attr: Attribute):
        """Reformat noodle attributes to modernGL attribute format"""

        info = {
            "name": f"in_{attr.semantic.lower()}",
            "components": FORMAT_MAP[attr.format].num_components
            # "type": ?
        }
        return info

    @staticmethod
    def construct_format_str(attributes: dict):
        """Helper to construct format string from Noodle Attribute dict
        
        Looking for str like "3f 3f" for interleaved positions and normals
        """

        formats = []
        norm_factor = None
        for attr in attributes:
            format_info = FORMAT_MAP[attr.format]
            formats.append(f"{format_info.num_components}{format_info.format}")

            # If texture is present, calculate number to divide by in vertex shader
            if attr.semantic == "TEXTURE":
                norm_factor = (2 ** (format_info.size * 8)) - 1

        return " ".join(formats), norm_factor

    def render(self, instances, window):

        # Render each patch using the instances
        nodes = []
        num_instances = 0
        for patch in self.patches:
            node, num_instances = self.render_patch(patch, instances, window)
            nodes.append(node)
        return nodes, num_instances

    def render_patch(self, patch, instances, window):

        scene = window.scene

        # Initialize VAO to store buffers and indices for this patch
        vao = mglw.opengl.vao.VAO(name=f"{self.name} Patch VAO", mode=MODE_MAP[patch.type])

        # Get Material - for now material delegate uses default texture
        material = self.client.get_component(patch.material)
        scene.materials.append(material.mglw_material)

        # Reformat attributes
        noodle_attributes = patch.attributes
        new_attributes = {attr.semantic: GeometryDelegate.reformat_attr(attr) for attr in noodle_attributes}

        # Get Index Bytes and Size to use later in vao
        index = patch.indices
        index_view = self.client.get_component(index.view)
        index_bytes = index_view.buffer_delegate.bytes[index.offset:]
        index_size = FORMAT_MAP[index.format].size
        vao.index_buffer(index_bytes, index_size)

        # Construct vertex array object from attribute buffers and buffer views
        buffer_groups = {}
        for attribute in patch.attributes:
            buffer_groups.setdefault(attribute.view, []).append(attribute)

        # Create a vertex array buffer for each group of attributes
        for view_id, attributes in buffer_groups.items():

            # Get Bytes from the view
            vertex_view = self.client.get_component(view_id)
            buffer = vertex_view.buffer_delegate
            vertex_bytes = buffer.bytes[:index.offset] if index_view == vertex_view else buffer.bytes

            # Format attributes for mglw, and add to vao
            buffer_format, norm_factor = GeometryDelegate.construct_format_str(noodle_attributes)
            attr_names = [new_attributes[attribute.semantic]["name"] for attribute in attributes]
            vao.buffer(vertex_bytes, buffer_format, attr_names)

        # Add default attributes for those that are missing
        if "COLOR" not in new_attributes:
            default_colors = [1.0, 1.0, 1.0, 1.0] * patch.vertex_count
            buffer_data = np.array(default_colors, np.single)
            vao.buffer(buffer_data, '4f', 'in_color')

        if "NORMAL" not in new_attributes:
            default_normal = [0.0, 0.0, 0.0] * patch.vertex_count
            buffer_data = np.array(default_normal, np.single)
            vao.buffer(buffer_data, '3f', 'in_normal')

        if "TEXTURE" not in new_attributes:
            default_texture_coords = [0.0, 0.0] * patch.vertex_count
            buffer_data = np.array(default_texture_coords, np.single)
            vao.buffer(buffer_data, '2f', 'in_texture')
            norm_factor = (2 ** (FORMAT_MAP["VEC2"].size * 8)) - 1

        # Create Mesh
        mesh = mglw.scene.Mesh(f"{self.name} Mesh", vao=vao, material=material.mglw_material, attributes=new_attributes)
        mesh.norm_factor = norm_factor

        # Add instances to vao if applicable, also add appropriate mesh program
        if instances:
            instance_view = self.client.get_component(instances.view)
            instance_buffer = instance_view.buffer_delegate
            instance_bytes = instance_buffer.bytes
            vao.buffer(instance_bytes, '16f/i', 'instance_matrix')

            num_instances = int(instance_buffer.size / 64)  # 16 4 byte floats per instance
            mesh.mesh_program = programs.PhongProgram(window, num_instances)

            # For debugging, instances...
            # instance_list = np.frombuffer(instance_bytes, np.single).tolist()
            # positions = []
            # rotations = []
            # for i in range(num_instances):
            #     j = 16 * i
            #     positions.append(instance_list[j:j+3])
            #     rotations.append(instance_list[j+8:j+12])
            # print(f"Instance rendering positions: \n{positions}")
            # print(f"Instance rendering rotations: \n{rotations}")

        else:
            num_instances = 0
            mesh.mesh_program = programs.PhongProgram(window, num_instances=-1)

        # Add mesh as new node to scene graph
        scene.meshes.append(mesh)
        new_mesh_node = mglw.scene.Node(self.name, mesh=mesh, matrix=np.identity(4))
        root = scene.root_nodes[0]
        new_mesh_node.matrix_global = root.matrix_global
        root.add_child(new_mesh_node)
        window.scene.nodes.append(new_mesh_node)
        return new_mesh_node, num_instances

    # def on_new(self, message: dict):

        # assuming all attrs use same view
        # first_patch_attrs = self.patches[0].attributes
        # view_id = first_patch_attrs[0].view
        # self.buffer_view = self.client.get_component(view_id)

    def on_remove(self, message: dict):
        pass

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
        index_view = self.client.get_component(index.view)
        index_view.gui_rep()
        imgui.text(f"Count: {index.count}")
        imgui.text(f"Offset: {index.offset}")
        imgui.text(f"Stride: {index.stride}")
        imgui.text(f"Format: {index.format}")
        imgui.separator()

        if patch.material:
            self.client.get_component(patch.material).gui_rep()

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
        self.texture_delegate = self.client.get_component(self.pbr_info.base_color_texture.texture)
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
    component_map = {
        "RGB": 3,
        "RGBA": 4
    }

    def on_new(self, message: dict):

        # Get Bytes from either source present
        if self.buffer_source:
            buffer = self.client.get_component(self.buffer_source)
            im = img.open(io.BytesIO(buffer.bytes))
            im = im.transpose(img.FLIP_TOP_BOTTOM)
            self.size = im.size
            self.components = self.component_map[im.mode]
            self.bytes = im.tobytes()
        else:
            with urllib.request.urlopen(self.uri_source) as response:
                self.bytes = response.read()

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
        image = self.client.get_component(self.image)
        self.image_delegate = image
        self.mglw_texture = window.ctx.texture(image.size, image.components, image.bytes)
        self.image_delegate.texture_id = self.mglw_texture.glo

    def on_new(self, message: dict):

        self.client.callback_queue.put((self.set_up_texture, []))

        if self.sampler:
            self.sampler_delegate = self.client.get_component(self.sampler)

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

    FILTER_MAP = {
        "NEAREST": moderngl.NEAREST,
        "LINEAR": moderngl.LINEAR,
        "LINEAR_MIPMAP_LINEAR": moderngl.LINEAR_MIPMAP_LINEAR,
    }

    SAMPLER_MODE_MAP = {
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
        self.buffer_delegate: BufferDelegate = self.client.get_component(self.source_buffer)

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
