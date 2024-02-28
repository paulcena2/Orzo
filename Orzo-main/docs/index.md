# Overview

![Build Status](https://github.com/InsightCenterNoodles/Orzo/workflows/CI/badge.svg)
![PyPI](https://img.shields.io/pypi/v/Orzo)
[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/InsightCenterNoodles/Orzo/python-coverage-comment-action-data/endpoint.json&color=red)](https://htmlpreview.github.io/?https://github.com/InsightCenterNoodles/Orzo/blob/python-coverage-comment-action-data/htmlcov/index.html)

Orzo is the first client application to build on the NOODLES protocol to render 3d scenes in Python. NOODLES allows multiple client
applications to interact collaboratively with data in real-time, and now these scenes can be easily displayed in Python.
Orzo uses moderngl and Phong shading to bring complex geometry to life.

<video autoplay loop src="assets/demo.mov">  video </video> 


## Why use Orzo?

Orzo is a great choice for anyone who wants to easily visualize a NOODLES scene using Python. Whether you are debugging
and inspecting state or joining a collaborative session, Orzo offers an accessible entry point for using NOODLES.
You can easily inspect the state of the scene and invoke methods from a simple GUI. There is also built
in support for moving and manipulating mesh objects in the scene. Entities can be dragged, rotated, and scaled directly or 
with configurable widgets. There are also options to change the skybox, default lighting, and shader settings.
With Orzo, you can connect to any server that implements the Noodles protocol. If you are looking to use NOODLES with 
a different language, there is currently support for C++, Rust, 
Julia, and Javascript [here](https://github.com/InsightCenterNoodles/). This library is build on top of the 
[Penne](https://insightcenternoodles.github.io/Penne/) library, and if you are looking to build a server in Python
to connect with, check out [Rigatoni](https://insightcenternoodles.github.io/Rigatoni/).

## Getting Started
1. Install the library
```bash
pip install orzo
```
2. Start running and create a window
```python
import orzo

orzo.run("ws://localhost:50000")

```