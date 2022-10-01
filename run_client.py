import penne
from tagliatelle.delegates import *


def start_render():
    pass


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
    client = penne.create_client("ws://localhost:50000", del_hash, on_connected=start_render)

if __name__ == "__main__":
    start()