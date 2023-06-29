import logging

import penne

from .delegates import delegate_map
from .window import Window


def run(address="ws://localhost:50000", default_lighting=True, on_connected=None):
    """Runs the Orzo Client

    Connects to address and opens a window to display the scene. The window will run
    indefinitely until it is closed. If window is black with just the gui displayed,
    it is possible that the server does not have any lights in the scene. The default
    lighting parameter is set to True by default, and helps avoid this situation, but
    it can be turned off if desired.

    Args:
        address (str): Address of the server to connect to
        default_lighting (bool): Whether to use default lighting
        on_connected (function): Function to run when connected to server
    """

    # Update forward refs where entity -> light -> client -> entity
    for delegate in delegate_map.values():
        delegate.update_forward_refs()

    # Create Client and start rendering loop
    with penne.Client(address, delegate_map, on_connected) as render_client:
        Window.client = render_client
        Window.default_lighting = default_lighting
        Window.run()  # Runs indefinitely until window is closed

    logging.info(f"Finished Running Orzo Client")


class Client(object):
    """Mock up for client context manager

    Currently, doesn't work because of the way the window is set up - run blocks, and do we work with instance or cls
    Would this be helpful, and do people want access to the window / client instance? Thinking of noodles explorer,
    maybe not. Might be nice for testing though, could run and then stop it using script instead of gui. Would we need
    it to be fully running for testing, or should I focus on the delegate methods and stuff that is more independent.
    """

    def __init__(self, address="ws://localhost:50000", default_lighting=True, on_connected=None):
        self.address = address
        self.default_lighting = default_lighting
        self.on_connected = on_connected

    def __enter__(self):

        # Update forward refs where entity -> light -> client -> entity
        for delegate in delegate_map.values():
            delegate.update_forward_refs()

        # Create Client
        render_client = penne.Client(self.address, delegate_map, self.on_connected)
        render_client.thread.start()
        render_client.connection_established.wait()

        # Create Window and start rendering loop
        self.window = Window
        self.window.client = render_client
        self.window.default_lighting = self.default_lighting
        self.window.run()  # Runs indefinitely until window is closed
        return self.window

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.window.client.shutdown()
        self.window.close()
        logging.info(f"Finished Running Orzo Client")


