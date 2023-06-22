import logging

import penne

from orzo.delegates import *
from orzo.window import Window

del_hash = {
    "entities": EntityDelegate,
    "tables": TableDelegate,
    "plots": PlotDelegate,
    "signals": SignalDelegate,
    "methods": MethodDelegate,
    "materials": MaterialDelegate,
    "geometries": GeometryDelegate,
    "lights": LightDelegate,
    "images": ImageDelegate,
    "textures": TextureDelegate,
    "samplers": SamplerDelegate,
    "buffers": BufferDelegate,
    "bufferviews": BufferViewDelegate,
    "document": DocumentDelegate
}

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG
)


def run(default_lighting=True):

    # Update forward refs where entity -> light -> client -> entity
    for delegate in del_hash.values():
        delegate.update_forward_refs()

    # Create Client and start rendering loop
    with penne.Client("ws://localhost:50000", del_hash) as render_client:
        Window.client = render_client
        Window.default_lighting = default_lighting
        Window.run()  # Runs indefinitely until window is closed

    logging.info(f"Finished Running Tagliatelle Client")


if __name__ == "__main__":
    run()
