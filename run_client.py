import penne

from tagliatelle.delegates import *
from tagliatelle.window import Window


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

    # Create Client and start rendering loop
    client = penne.create_client("ws://localhost:50000", del_hash)
    Window.client = client
    Window.run()

    # Shutdown and wait for client thread to finish
    client.shutdown()
    print(f"Finished Testing")


if __name__ == "__main__":
    start()