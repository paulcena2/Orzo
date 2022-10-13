
import penne
import moderngl
import moderngl_window as mglw
import multiprocessing
import numpy as np

from tagliatelle.delegates import *
from tagliatelle.windows import *


def launch():
    sender, receiver = multiprocessing.Pipe()
    # process = multiprocessing.Process(target=GridWindow.run, args=())
    process = multiprocessing.Process(target=CameraWindow.run, args=())
    process.start()


# def render(receiver):

#     # main = threading.main_thread()
#     # print(main)
#     # asyncio.run_coroutine_threadsafe(mglw.run_window_config(Test), main)
#     mglw.run_window_config(GridWindow)
#     sphere = mglw.geometry.sphere()
#     print(f"Sphere: {sphere}")
#     while True:
#         if receiver.poll(.1):
#             update = receiver.recv()
#             print(f"Update to render: {update}")


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
    client = penne.create_client("ws://localhost:50000", del_hash, on_connected=launch)
    client.thread.join()

if __name__ == "__main__":
    start()