import queue

import penne
import moderngl
import moderngl_window as mglw
import multiprocessing
import numpy as np

from tagliatelle.delegates import *
from tagliatelle.windows import *


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

    # Wait for client thread to finish
    client.thread.join()
    print(f"Finished Testing")


if __name__ == "__main__":
    start()