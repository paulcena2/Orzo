import penne
import moderngl
import moderngl_window as mglw

from tagliatelle.delegates import *

class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Do initialization here
        print(type(self.ctx))
        # self.prog = self.ctx.program(...) # create program in moderngl context for shaders
        # self.vao = self.ctx.vertex_array(...)
        # self.texture = self.ctx.texture(self.wnd.size, 4)

    def render(self, time, frametime):
        # This method is called every frame
        # self.vao.render()
        pass

    def resize(self, width: int, height: int):
        print("Window was resized. buffer size is {} x {}".format(width, height))

    def close(self):
        print("The window is closing")


def start_render():
    mglw.run_window_config(Test)


del_hash = {
    "entities" : EntityDelegate,
    "tables" : TableDelegate,
    "plots" : PlotDelegate,
    "signals" : SignalDelegate,
    "methods" : MethodDelegate,
    "materials" : MaterialDelegate,
    "geometries" : GeometryDelegate,
    "lights" : LightDelegate,
    "images" : ImageDelegate,
    "textures" : TextureDelegate,
    "samplers" : SamplerDelegate,
    "buffers" : BufferDelegate,
    "bufferviews" : BufferViewDelegate,
    "document" : DocumentDelegate
}

def start():
    client = penne.create_client("ws://localhost:50000", del_hash)
    start_render()
    client.thread.join()

if __name__ == "__main__":
    start()