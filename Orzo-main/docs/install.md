Installation is as simple as:

```bash
pip install orzo
```

Orzo has a few dependencies:

* [`penne`](https://insightcenternoodles.github.io/Penne/): NOODLES client library
* [`pyglet`](https://pyglet.readthedocs.io/en/latest/): Window backend and event handling for OpenGL
* [`moderngl`](https://moderngl.readthedocs.io/en/latest/): Modern OpenGL wrapper for rendering
* [`moderngl-window`](https://moderngl-window.readthedocs.io/en/latest/): Windowing and event handling for OpenGL
* [`imgui`](https://pyimgui.readthedocs.io/en/latest/): Immediate mode GUI for Python
* [`numpy`](https://numpy.org/doc/stable/): Array manipulation for Python, used in delegates
* [`numpy-quaternion`](https://quaternion.readthedocs.io/en/latest/Package%20API%3A/quaternion/numpy_quaternion.cpython-38-x86_64-linux-gnu/): Quaternion manipulation for Python, used for entity rotations
* [`Pillow`](https://pillow.readthedocs.io/en/stable/): Image manipulation for Python, used for texture loading

If you've got Python 3.9+ and `pip` installed, you're good to go.

!!! Note

    For stability, Rigatoni's core dependencies are pinned to specific versions. While these are up to date as of August
    2023, you may want to update them to the latest versions. To do so, simply update the package yourself.

