import logging

import penne

from .delegates import delegate_map
from .window import Window


def run(address, default_lighting=True):
    """Runs the Orzo Client

    Connects to address and opens a window to display the scene. The window will run
    indefinitely until it is closed. If window is black with just the gui displayed,
    it is possible that the server does not have any lights in the scene. The default
    lighting parameter is set to True by default, and helps avoid this situation, but
    it can be turned off if desired.

    Args:
        address (str): Address of the server to connect to
        default_lighting (bool): Whether or not to use default lightin
    """

    # Update forward refs where entity -> light -> client -> entity
    for delegate in delegate_map.values():
        delegate.update_forward_refs()

    # Create Client and start rendering loop
    with penne.Client(address, delegate_map) as render_client:
        Window.client = render_client
        Window.default_lighting = default_lighting
        Window.run()  # Runs indefinitely until window is closed

    logging.info(f"Finished Running Orzo Client")
