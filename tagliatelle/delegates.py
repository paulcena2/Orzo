"""Module Defining Default Delegates and Delegate Related Classes"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from urllib.parse import _NetlocResultMixinStr
if TYPE_CHECKING:
    from penne.messages import Message
    from penne.core import Client

from . import programs

from penne import Delegate, inject_methods, inject_signals
import moderngl_window as mglw
import moderngl
import numpy as np


@dataclass
class FormatInfo:
    num_components: int
    format: str
    size: int # in bytes


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
    "TRIANGLES" : moderngl.TRIANGLES,
    "POINTS" : moderngl.POINTS,
    "LINES" : moderngl.LINES,
    "LINE_LOOP" : moderngl.LINE_LOOP,
    "LINE_STRIP" : moderngl.LINE_STRIP,
    "TRIANGLE_STRIP" : moderngl.TRIANGLE_STRIP
}


class MethodDelegate(Delegate):
    """Delegate representing a method which can be invoked on the server

    Attributes:
        client (client object): 
            client delegate is a part of 
        info (message): 
            message containing information on the method
        specifier (str): 
            keyword for specifying the type of delegate
        context_map (dict):
            mapping specifier to context for method invocation
    """

    def __init__(self, client: Client, message: Message, specifier: str):
        self.client = client
        self.info = message
        self.specifier = specifier
        self.context_map = {
            "tables": "table",
            "plots": "plot",
            "entities": "entity"
        }

    def on_new(self, message: Message):
        pass

    def on_remove(self, message: Message):
        pass

    def invoke(self, on_delegate: Delegate, args=None, callback=None):
        """Invoke this delegate's method

        Args:
            on_delegate (delegate):
                delegate method is being invoked on 
                used to get context
            args (list, optional):
                args for the method
            callback (function):
                function to be called when complete
        """
        context = {self.context_map[on_delegate.specifier]: on_delegate.info.id}
        self.client.invoke_method(self.info.id, args, context=context, on_done=callback)


    def __repr__(self) -> str:
        """Custom string representation for methods"""
        
        rep = f"{self.info.name}:\n\t{self.info.doc}\n\tReturns: {self.info.return_doc}\n\tArgs:"
        for arg in self.info.arg_doc:
            rep += f"\n\t\t{arg.name}: {arg.doc}"
        return rep


class SignalDelegate(Delegate):
    """Delegate representing a signal coming from the server

    Attributes:
        client (Client): 
            client delegate is a part of 
        info (message): 
            message containing information on the signal
        specifier (str): 
            keyword for specifying the type of delegate
    """
    
    def __init__(self, client: Client, message: Message, specifier: str):
        self.client = client
        self.info = message
        self.specifier = specifier

    def on_new(self, message: Message):
        pass

    def on_remove(self, message: Message): 
        pass


class SelectionRange(tuple):
    """Selection of range of rows"""

    def __new__(cls, key_from: int, key_to: int):
        return super().__new__(SelectionRange, (key_from, key_to))


class Selection(object):
    """Selection of certain rows in a table

    Attributes:
        name (str): 
            name of the selection
        rows (list[int]): 
            list of indices of rows
        row_ranges (list[SelectionRange]): 
            ranges of selected rows
    """

    def __init__(self, name: str, rows: list[int] = None, row_ranges: list[SelectionRange] = None):
        self.name = name
        self.rows = rows
        self.row_ranges = row_ranges

    def __repr__(self) -> str:
        return f"Selection Object({self.__dict__})"

    def __getitem__(self, attribute):
        return getattr(self, attribute)


class TableDelegate(Delegate):
    """Delegate representing a table

    Each table delegate corresponds with a table on the server
    To use the table, you must first subscribe 

    Attributes:
        client (Client): 
            weak ref to client to invoke methods and such
        dataframe (Dataframe): 
            dataframe representing current state of the table
        selections (dict): 
            mapping of name to selection object
        signals (signals): 
            mapping of signal name to function
        name (str): 
            name of the table
        id (list): 
            id group for delegate in state and table on server
    """

    def __init__(self, client: Client, message: Message, specifier: str):
        super().__init__(client, message, specifier)
        self.name = "Table Delegate"
        self.selections = {}
        self.signals = {
            "tbl_reset" : self._reset_table,
            "tbl_rows_removed" : self._remove_rows,
            "tbl_updated" : self._update_rows,
            "tbl_selection_updated" : self._update_selection
        }
        # Specify public methods 
        self.__all__ = [
            "subscribe", 
            "request_clear", 
            "request_insert", 
            "request_remove", 
            "request_update", 
            "request_update_selection",
            "plot"
        ]


    def _on_table_init(self, init_info: Message, on_done=None):
        """Creates table from server response info

        Args:
            init_info (Message Obj): 
                Server response to subscribe which has columns, keys, data, 
                and possibly selections
        """

        # Extract data from init info and transpose rows to cols
        row_data = getattr(init_info, "data")
        cols = getattr(init_info, "columns")
        print(f"Table Initialized with cols: {cols} and row data: {row_data}")


    def _reset_table(self):
        """Reset dataframe and selections to blank objects

        Method is linked to 'tbl_reset' signal
        """

        self.selections = {}


    def _remove_rows(self, key_list: list[int]):
        """Removes rows from table

        Method is linked to 'tbl_rows_removed' signal

        Args:
            key_list (list): list of keys corresponding to rows to be removed
        """

        print(f"Removed Rows: {key_list}...\n", self.dataframe)



    def _update_rows(self, keys: list[int], rows: list):
        """Update rows in table

        Method is linked to 'tbl_updated' signal

        Args:
            keys (list): 
                list of keys to update
            cols (list): 
                list of cols containing the values for each new row,
                should be col for each col in table, and value for each key
        """

        print(f"Updated Rows...{keys}\n")
        

    def _update_selection(self, selection_obj: Selection):
        """Change selection in delegate's state to new selection object

        Method is linked to 'tbl_selection_updated' signal

        Args:
            selection_obj (Selection): 
                obj with new selections to replace obj with same name
        """

        self.selections[selection_obj.name] = selection_obj
        print(f"Made selection {selection_obj.name} = {selection_obj}")


    def _relink_signals(self):
        """Relink the signals for built in methods

        These should always be linked, along with whatever is injected,
        so relink on new and on update messages
        """

        self.signals["noo::tbl_reset"] = self._reset_table
        self.signals["noo::tbl_rows_removed"] = self._remove_rows
        self.signals["noo::tbl_updated"] = self._update_rows
        self.signals["noo::tbl_selection_updated"] = self._update_selection


    def on_new(self, message: Message):
        """Handler when create message is received

        Args:
            message (Message): create message with the table's info
        """
        
        # Set name
        name = message["name"]
        methods = message["methods_list"]
        signals = message["signals_list"]
        if name: self.name = name
    
        # Inject methods and signals
        if methods: inject_methods(self, methods)
        if signals: inject_signals(self, signals)

        # Reset
        self._reset_table()
        self._relink_signals()


    def on_update(self, message: Message):
        """Handler when update message is received
        
        Args:
            message (Message): update message with the new table's info
        """

        self._relink_signals()
    

    def on_remove(self, message: Message):
        pass


    def subscribe(self, on_done: Callable=None):
        """Subscribe to this delegate's table

        Calls on_table_init as callback
        
        Raises:
            Exception: Could not subscribe to table
        """

        try:
            # Allow for calback after table init
            lam = lambda data: self._on_table_init(data, on_done)
            self.tbl_subscribe(on_done=lam)
        except:
            raise Exception("Could not subscribe to table")

    
    def request_insert(self, row_list: list[list[int]], on_done=None):
        """Add rows to end of table

        User endpoint for interacting with table and invoking method
        For input, row list is list of rows. Also note that tables have
        nine columns by default (x, y, z, r, g, b, sx, sy, sz).
        x, y, z -> coordinates
        r, g, b -> color values [0, 1]
        sx, sy, sz -> scaling factors, default size is 1 meter

        Row_list: [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

        Args:
            col_list (list, optional): add rows as list of columns
            row_list (list, optional): add rows using list of rows
            on_done (function, optional): callback function
        Raises:
            Invalid input for request insert exception
        """

        self.tbl_insert(on_done, row_list)
    

    def request_update(self, keys:list[int], rows:list[list[int]], on_done=None):
        """Update the table using a DataFrame

        User endpoint for interacting with table and invoking method

        Args:
            data_frame (DataFrame):
                data frame containing the values to be updated
            on_done (function, optional): 
                callback function called when complete
        """
        
        self.tbl_update(on_done, keys, rows)


    def request_remove(self, keys: list[int], on_done=None):
        """Remove rows from table by their keys

        User endpoint for interacting with table and invoking method

        Args:
            keys (list):
                list of keys for rows to be removed
            on_done (function, optional): 
                callback function called when complete
        """

        self.tbl_remove(on_done, keys)


    def request_clear(self, on_done=None):
        """Clear the table

        User endpoint for interacting with table and invoking method

        Args:
            on_done (function, optional): callback function called when complete
        """
        self.tbl_clear(on_done)


    def request_update_selection(self, name: str, keys: list[int], on_done=None):
        """Update a selection object in the table

        User endpoint for interacting with table and invoking method

        Args:
            name (str):
                name of the selection object to be updated
            keys (list):
                list of keys to be in new selection
            on_done (function, optional): 
                callback function called when complete
        """

        self.tbl_update_selection(on_done, name, {"rows": keys})


class DocumentDelegate(Delegate):
    pass


class EntityDelegate(Delegate):

    def __init__(self, client: Client, message: Message, specifier: str):
        super().__init__(client, message, specifier)
        self.name = "No-Name Entity" if not hasattr(self.info, "name") else self.info.name

    def render_entity(self, window):
        
        # Prepare Mesh
        render_rep = self.info.render_rep
        geometry = self.client.state["geometries"][render_rep["mesh"]]
        patches = geometry.patches
        instances = render_rep.instances if hasattr(render_rep, "instances") else None
        
        # Render Each Patch Using Geometry Delegate
        for patch in patches:
            geometry.render_patch(patch, instances, window)

    def attach_lights(self, window):

        for light_id in self.info.lights:

            # Add Positiona and direction to info
            light_delegate = self.client.get_component("lights", tuple(light_id))
            light_info = light_delegate.light_basics
            world_transform = self.get_world_transform()
            world_pos = np.matmul(world_transform, np.array([0.0, 0.0, 0.0, 1.0]))
            direction = np.matmul(world_transform, np.array([0.0, 0.0, -1.0, 1.0]))
            light_info["world_position"] = (world_pos[0]/world_pos[3], world_pos[1]/world_pos[3], world_pos[2]/world_pos[3])
            light_info["direction"] = (direction[0]/direction[3], direction[1]/direction[3], direction[2]/direction[3])
        
            # Update State
            id = light_delegate.info.id
            if id not in window.lights:
                window.lights[id]= light_info
                window.num_lights += 1


    def get_world_transform(self):

        # Swap axis to go from col major -> row major order
        local_transform = np.array(self.info.transform).reshape(4, 4).swapaxes(0, 1)

        if not hasattr(self.info, "parent"):
            return local_transform

        else:
            parent = self.client.get_component("entities", self.info.parent)
            return np.matmul(parent.get_world_transform(), local_transform)


    def on_new(self, message: Message):
       
        if hasattr(self.info, "render_rep"):
            self.client.callback_queue.put((self.render_entity, []))

        if hasattr(self.info, "lights"):
            self.client.callback_queue.put((self.attach_lights, []))


    def on_remove(self, message: Message):
        
        self.client.callback_queue.put((self.remove_from_render, []))


class PlotDelegate(Delegate):
    pass


class MaterialDelegate(Delegate):
    
    def on_new(self, message: Message):
        """"
        MsgMaterialCreate = {
            id : MaterialID,
            ? name : tstr,

            ? pbr_info : PBRInfo, ; if missing, defaults
            ? normal_texture : TextureRef,
            
            ? occlusion_texture : TextureRef, ; assumed to be linear, ONLY R used
            ? occlusion_texture_factor : float, ; assume 1 by default

            ? emissive_texture : TextureRef, ; assumed to be SRGB. ignore A.
            ? emissive_factor  : Vec3, ; all 1 by default

            ? use_alpha    : bool,  ; false by default
            ? alpha_cutoff : float, ; .5 by default

            ? double_sided : bool, ; false by default
        }
        """
        self.name = "No-Name Material" if not hasattr(message, "name") else message.name

        material = mglw.scene.Material(f"{self.name}")
        material.color = message.pbr_info.base_color
        if hasattr(message, "normal_texture"):
            texture = self.client.get_component("textures", message.normal_texture.texture)
            material.mat_texture = mglw.scene.MaterialTexture(texture.mglw_texture, texture.mglw_sampler)
    
        material.double_sided = False if not hasattr(message, "double_sided") else message.double_sided
        self.mglw_material = material


class GeometryDelegate(Delegate):

    def reformat_attr(self, attr: dict):
        """Reformat noodle attributes to modernGL attribute format"""

        info = {
            "name": f"in_{attr['semantic'].lower()}",
            "components": FORMAT_MAP[attr['format']].num_components
            #"type": ?
        }
        return info


    def construct_format_str(self, attributes: dict):
        """Helper to construct format string from Noodle Attribute dict
        
        Looking for str like "3f 3f" for interleaved positions and normals
        """

        formats = []
        for attr in attributes:
            format_info = FORMAT_MAP[attr["format"]]
            formats.append(f"{format_info.num_components}{format_info.format}")
        return " ".join(formats)


    def render_patch(self, patch, instances, window):
        
        scene = window.scene

        # Get Material - TODO: convert from noodles to mglw
        material = self.client.get_component("materials", patch.material)
        scene.materials.append(material.mglw_material)

        # Reformat attributes
        noodle_attributes = patch.attributes
        new_attributes = {attr.semantic: self.reformat_attr(attr) for attr in noodle_attributes}

        # Construct vertex array object from buffer and buffer view
        view = self.buffer_view
        buffer = view.buffer
        index_offset = patch.indices["offset"] 
        buffer_format = self.construct_format_str(noodle_attributes)
        vao = mglw.opengl.vao.VAO(name=f"{self.name} Patch VAO", mode=MODE_MAP[patch['type']])
        vao.buffer(buffer.bytes[:index_offset], buffer_format, [info["name"] for info in new_attributes.values()])
        index_bytes, index_size = buffer.bytes[index_offset:], FORMAT_MAP[patch.indices["format"]].size
        vao.index_buffer(index_bytes, index_size)

        # Add default attributes for those that are missing
        if "COLOR" not in new_attributes:
            default_colors = [1.0, 1.0, 1.0, 1.0] * patch['vertex_count']
            buffer_data = np.array(default_colors, np.single)
            vao.buffer(buffer_data, '4f', 'in_color')

        if "NORMAL" not in new_attributes:
            default_normal = [0.0, 0.0, 0.0] * patch['vertex_count']
            buffer_data = np.array(default_normal, np.single)
            vao.buffer(buffer_data, '3f', 'in_normal')

        if "TEXTURE" not in new_attributes:
            default_texture_coords = [0.0, 0.0] * patch['vertex_count']
            buffer_data = np.array(default_texture_coords, np.single)
            vao.buffer(buffer_data, '2f', 'in_color')
    
        # Create Mesh and add rendering program
        mesh = mglw.scene.Mesh(f"{self.name} Mesh", vao=vao, material=material.mglw_material, attributes=new_attributes)
        mesh.lights = window.lights
        mesh.num_lights = window.num_lights
        
        # Add instances to vao if applicable
        if instances:
            instance_view = self.client.state["bufferviews"][instances.view]
            instance_buffer = instance_view.buffer
            instance_bytes = instance_buffer.bytes
            num_instances = int(instance_buffer.size / 64) # 16 4 byte floats per instance
            print(f"Instance Bytes: \n{np.frombuffer(instance_bytes, dtype=np.single)}")

            vao.buffer(instance_bytes, '16f/i', 'instance_matrix')

            # mesh.mesh_program = programs.InstanceProgram(window.ctx, num_instances)
            mesh.mesh_program = programs.PhongProgram(window.ctx, num_instances)
            instance_list = np.frombuffer(instance_bytes, np.single).tolist()
            positions = []
            rotations = []
            for i in range(num_instances):
                j = 16 * i
                positions.append(instance_list[j:j+3])
                rotations.append(instance_list[j+8:j+12])
            print(f"Instance rendering positions: \n{positions}")
            print(f"Instance rendering rotations: \n{rotations}")

        else:
            mesh.mesh_program = programs.BaseProgram(window.ctx)
        scene.meshes.append(mesh)
        
        # Add mesh as new node to scene graph
        new_mesh_node = mglw.scene.Node(self.name, mesh=mesh)
        root = scene.root_nodes[0]
        new_mesh_node.matrix_global = root.matrix_global # Is this how matrices should work here?
        root.add_child(new_mesh_node)
        window.scene.nodes.append(new_mesh_node)
        self.nodes.append(new_mesh_node)


    def remove_from_render(self, window):
        # Need to test, enough to remove from render?
        for node in self.nodes:
            window.scene.root_nodes[0].children.remove(node)
            window.scene.nodes.remove(node)

    
    def on_new(self, message: Message):

        self.name = "No-Name Geometry" if not hasattr(message, "name") else message.name
        self.patches = message.patches
        self.first_patch_attrs = self.patches[0].attributes
        self.nodes = []

        # assuming all attrs use same view
        view_id = self.first_patch_attrs[0]["view"] 
        self.buffer_view = self.client.state["bufferviews"][view_id]


    def on_remove(self, message: Message):
        self.client.callback_queue.put((self.remove_from_render, []))


class LightDelegate(Delegate):

    def get_light_type(self, message):
        if hasattr(message, "point"): return 0
        elif hasattr(message, "spot"): return 1
        elif hasattr(message, "directional"): return 2

    def format_color(self, color):
        formatted = [*color]
        if len(formatted) == 3:
            formatted.append(1.0)
        if color[0] > 1 or color[1] > 1 or color[2] > 1:
            for i in range(3):
                formatted[i] /= 255
        return tuple(formatted)

    def on_new(self, message: Message):
        
        # Add info based on light type
        light_type = self.get_light_type(message)
        color = self.format_color(message.color) if hasattr(message, "color") else (1.0, 1.0, 1.0, 10)
        intensity = message.intensity if hasattr(message, "intensity") else 1
        if light_type == 0:
            info = (intensity, message.point.range, 0.0, 0.0)
        elif light_type == 1:
            spot_info = message.spot
            info = (intensity, spot_info.range, spot_info.inner_cone_angle_rad, spot_info.outer_cone_angle_rad)
        else:
            info = (intensity, message.directional.range, 0.0, 0.0)

        # Arrange info into dict to store
        self.light_basics = {
            "color": color,
            "ambient": (.1, .1, .1),
            "type": light_type,
            "info": info,
        }


class ImageDelegate(Delegate):
    
    def on_new(self, message: Message):
        self.size = (0, 0)
        self.components = 1
        if hasattr(message, "buffer_source"):
            buffer = self.client.get_component("bufferviews", message.buffer_source).buffer
            self.bytes = buffer.bytes
        else:
            print("URI source not implemented for images")
            self.bytes = b'Uh Oh'


class TextureDelegate(Delegate):

    def set_up_texture(self, window):
        image = self.client.get_component("images", self.info.image)
        self.mglw_texture = window.ctx.texture(image.size, image.components, image.bytes) # Need size (width, height), components, data - bytes,

    def set_up_sampler(self, window):
        if hasattr(self.info, "sampler"):
            sampler = self.client.get_component("samplers", self.info.sampler)
            self.mglw_sampler = sampler.mglw_sampler
        else:
            self.mglw_sampler = window.ctx.sampler(texture=self.mglw_texture)
    
    def on_new(self, message: Message):
        self.mglw_texture = None        
        self.client.callback_queue.put((self.set_up_texture, []))
        self.client.callback_queue.put((self.set_up_sampler, []))
        

class SamplerDelegate(Delegate):

    FILTER_MAP = {
        "NEAREST": moderngl.NEAREST,
        "LINEAR": moderngl.LINEAR,
        "LINEAR_MIPMAP_LINEAR": moderngl.LINEAR_MIPMAP_LINEAR 
    }

    def set_up_sampler(self, window):

        # yet to include wrap_s, wrap_t, -> repeat_x repeat_y in moderngl?
        min_filter = self.FILTER_MAP[self.info.min_filter] if hasattr(self.info, "min_filter") else moderngl.LINEAR_MIPMAP_LINEAR
        mag_filter = self.FILTER_MAP[self.info.mag_filter] if hasattr(self.info, "mag_filter") else moderngl.LINEAR
        self.mglw_sampler = window.ctx.sampler(filter=(min_filter, mag_filter))
    
    def on_new(self, message: Message):
        self.mglw_sampler = None
        self.client.callback_queue.put((self.set_up_sampler, []))


class BufferDelegate(Delegate):  

    def on_new(self, message: Message):
        self.size = message.size

        if hasattr(message, "inline_bytes"):
            self.bytes = message.inline_bytes
        else:
            print("URI Bytes Not Implemented Yet")
            self.bytes = b'Uh oh'


class BufferViewDelegate(Delegate):
    
    def on_new(self, message: Message):
        self.buffer: BufferDelegate = self.client.state["buffers"][message.source_buffer]

