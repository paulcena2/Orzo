import logging

import penne

from orzo.delegates import *
from orzo.window import Window

del_hash = {
    penne.Entity: EntityDelegate,
    penne.Table: TableDelegate,
    penne.Plot: PlotDelegate,
    penne.Signal: SignalDelegate,
    penne.Method: MethodDelegate,
    penne.Material: MaterialDelegate,
    penne.Geometry: GeometryDelegate,
    penne.Light: LightDelegate,
    penne.Image: ImageDelegate,
    penne.Texture: TextureDelegate,
    penne.Sampler: SamplerDelegate,
    penne.Buffer: BufferDelegate,
    penne.BufferView: BufferViewDelegate,
    penne.Document: DocumentDelegate
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

    logging.info(f"Finished Running Orzo Client")


if __name__ == "__main__":
    run()
