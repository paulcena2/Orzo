"""Test script for testing geometry creation library

Offers sample methods a server could implement using a sphere
"""

import asyncio
import rigatoni
from rigatoni import geometry as geo

# 42 vertices for sphere
vertices = [[-0.000000, -0.500000, -0.000000], [0.361804, -0.223610, 0.262863],
            [-0.138194, -0.223610, 0.425325], [-0.447213, -0.223608, -0.000000],
            [-0.138194, -0.223610, -0.425325], [0.361804, -0.223610, -0.262863],
            [0.138194, 0.223610, 0.425325], [-0.361804, 0.223610, 0.262863],
            [-0.361804, 0.223610, -0.262863], [0.138194, 0.223610, -0.425325],
            [0.447213, 0.223608, -0.000000], [-0.000000, 0.500000, -0.000000],
            [-0.081228, -0.425327, 0.249998], [0.212661, -0.425327, 0.154506],
            [0.131434, -0.262869, 0.404506], [0.425324, -0.262868, -0.000000],
            [0.212661, -0.425327, -0.154506], [-0.262865, -0.425326, -0.000000],
            [-0.344095, -0.262868, 0.249998], [-0.081228, -0.425327, -0.249998],
            [-0.344095, -0.262868, -0.249998], [0.131434, -0.262869, -0.404506],
            [0.475529, 0.000000, 0.154506], [0.475529, 0.000000, -0.154506],
            [-0.000000, 0.000000, 0.500000], [0.293893, 0.000000, 0.404508],
            [-0.475529, 0.000000, 0.154506], [-0.293893, 0.000000, 0.404508],
            [-0.293893, 0.000000, -0.404508], [-0.475529, 0.000000, -0.154506],
            [0.293893, 0.000000, -0.404508], [-0.000000, 0.000000, -0.500000],
            [0.344095, 0.262868, 0.249998], [-0.131434, 0.262869, 0.404506],
            [-0.425324, 0.262868, -0.000000], [-0.131434, 0.262869, -0.404506],
            [0.344095, 0.262868, -0.249998], [0.081228, 0.425327, 0.249998],
            [0.262865, 0.425326, -0.000000], [-0.212661, 0.425327, 0.154506],
            [-0.212661, 0.425327, -0.154506], [0.081228, 0.425327, -0.249998]]

# 80 triangles
indices = [[0, 13, 12], [1, 13, 15], [0, 12, 17], [0, 17, 19],
           [0, 19, 16], [1, 15, 22], [2, 14, 24], [3, 18, 26],
           [4, 20, 28], [5, 21, 30], [1, 22, 25], [2, 24, 27],
           [3, 26, 29], [4, 28, 31], [5, 30, 23], [6, 32, 37],
           [7, 33, 39], [8, 34, 40], [9, 35, 41], [10, 36, 38],
           [38, 41, 11], [38, 36, 41], [36, 9, 41], [41, 40, 11],
           [41, 35, 40], [35, 8, 40], [40, 39, 11], [40, 34, 39],
           [34, 7, 39], [39, 37, 11], [39, 33, 37], [33, 6, 37],
           [37, 38, 11], [37, 32, 38], [32, 10, 38], [23, 36, 10],
           [23, 30, 36], [30, 9, 36], [31, 35, 9], [31, 28, 35],
           [28, 8, 35], [29, 34, 8], [29, 26, 34], [26, 7, 34],
           [27, 33, 7], [27, 24, 33], [24, 6, 33], [25, 32, 6],
           [25, 22, 32], [22, 10, 32], [30, 31, 9], [30, 21, 31],
           [21, 4, 31], [28, 29, 8], [28, 20, 29], [20, 3, 29],
           [26, 27, 7], [26, 18, 27], [18, 2, 27], [24, 25, 6],
           [24, 14, 25], [14, 1, 25], [22, 23, 10], [22, 15, 23],
           [15, 5, 23], [16, 21, 5], [16, 19, 21], [19, 4, 21],
           [19, 20, 4], [19, 17, 20], [17, 3, 20], [17, 18, 3],
           [17, 12, 18], [12, 2, 18], [15, 16, 5], [15, 13, 16],
           [13, 0, 16], [12, 14, 2], [12, 13, 14], [13, 1, 14]]


def create_spheres(server: rigatoni.Server, context, *args):
    """Test method to create two spheres"""

    name = "Test Sphere"
    material = server.create_component(rigatoni.Material, name="Test Material")

    # Create Patch
    patches = []
    patch_info = geo.GeometryPatchInput(
        vertices=vertices,
        indices=indices,
        index_type="TRIANGLES",
        material=material.id
    )
    patches.append(geo.build_geometry_patch(server, name, patch_info))

    # Create geometry using patches
    sphere = server.create_component(rigatoni.Geometry, name=name, patches=patches)

    # Set instances and create an entity
    instances = geo.create_instances(
        positions=[[1, 1, 1, 1], [2, 2, 2, 2]],
        colors=[[1.0, .5, .5, 1.0]],
    )
    entity = geo.build_entity(server, geometry=sphere, instances=instances)

    # Add Lighting
    point_info = rigatoni.PointLight(range=-1)
    mat = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        3, 3, 3, 1
    ]
    light = server.create_component(rigatoni.Light, name="Test Point Light", point=point_info)
    # light2 = server.create_component(rigatoni.Light, name="Sun", intensity=5, directional=rigatoni.DirectionalLight())
    #server.create_component(rigatoni.Entity, transform=mat, lights=[light.id])

    spot_info = rigatoni.SpotLight()
    mat = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 3, 1
    ]
    spot = server.create_component(rigatoni.Light, name="Test Spot Light", spot=spot_info)
    #server.create_component(rigatoni.Entity, transform=mat, lights=[spot.id])

    direction_info = rigatoni.DirectionalLight()
    mat = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 5, 0, 1
    ]
    directional = server.create_component(rigatoni.Light, name="Test Spot Light", directional=direction_info)
    #server.create_component(rigatoni.Entity, transform=mat, lights=[directional.id])

    return 1


def delete_sphere(server: rigatoni.Server, context, *args):
    sphere = server.get_component_id(rigatoni.Entity, "Test Sphere")


    return 0


def move_sphere(server: rigatoni.Server, context, *args):
    # Change world transform, but do you change local?
    sphere = server.get_component_id(rigatoni.Entity, "Test Sphere")
    sphere.transform = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        args[0], args[1], args[2], 1
    ]
    server.update_component(sphere)
    pass


# Define starting state
starting_state = [
    rigatoni.StartingComponent(rigatoni.Method, {"name": "create_sphere", "arg_doc": []}, create_spheres),
    rigatoni.StartingComponent(rigatoni.Method, {"name": "move_sphere", "arg_doc": []}, move_sphere()),
    rigatoni.StartingComponent(rigatoni.Method, {"name": "delete_sphere", "arg_doc": []}, delete_sphere)
]


def run_test_server():
    asyncio.run(rigatoni.start_server(50000, starting_state))

