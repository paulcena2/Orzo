
import penne
import moderngl
import moderngl_window as mglw
import multiprocessing
import numpy as np

from tagliatelle.delegates import *

class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)
    title = "Test Window"
    aspect_ratio = 16 / 9
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Do initialization here
        print(type(self.ctx))
        # self.prog = self.ctx.program(...) # create program in moderngl context for shaders
        # self.vao = self.ctx.vertex_array(...)
        # self.texture = self.ctx.texture(self.wnd.size, 4)

        # START COPIED PROG - dont understand
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 vert;
                uniform float z_near;
                uniform float z_far;
                uniform float fovy;
                uniform float ratio;
                uniform vec3 center;
                uniform vec3 eye;
                uniform vec3 up;
                mat4 perspective() {
                    float zmul = (-2.0 * z_near * z_far) / (z_far - z_near);
                    float ymul = 1.0 / tan(fovy * 3.14159265 / 360);
                    float xmul = ymul / ratio;
                    return mat4(
                        xmul, 0.0, 0.0, 0.0,
                        0.0, ymul, 0.0, 0.0,
                        0.0, 0.0, -1.0, -1.0,
                        0.0, 0.0, zmul, 0.0
                    );
                }
                mat4 lookat() {
                    vec3 forward = normalize(center - eye);
                    vec3 side = normalize(cross(forward, up));
                    vec3 upward = cross(side, forward);
                    return mat4(
                        side.x, upward.x, -forward.x, 0,
                        side.y, upward.y, -forward.y, 0,
                        side.z, upward.z, -forward.z, 0,
                        -dot(eye, side), -dot(eye, upward), dot(eye, forward), 1
                    );
                }
                void main() {
                    gl_Position = perspective() * lookat() * vec4(vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 color;
                void main() {
                    color = vec4(0.04, 0.04, 0.04, 1.0);
                }
            ''',
        )

        self.prog['z_near'].value = 0.1
        self.prog['z_far'].value = 1000.0
        self.prog['ratio'].value = self.aspect_ratio
        self.prog['fovy'].value = 60

        self.prog['eye'].value = (3, 3, 3)
        self.prog['center'].value = (0, 0, 0)
        self.prog['up'].value = (0, 0, 1)
        # END COPIED PROG - dont understand 
        
        grid = []

        for i in range(65):
            grid.append([i - 32, -32.0, 0.0, i - 32, 32.0, 0.0])
            grid.append([-32.0, i - 32, 0.0, 32.0, i - 32, 0.0])

        grid = np.array(grid, dtype='f4')       

        self.vbo = self.ctx.buffer(grid)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'vert')

    def render(self, time: float, frametime: float):
        # This method is called every frame
        # self.vao.render()
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render(moderngl.LINES, 65 * 4)
        pass

    def resize(self, width: int, height: int):
        print("Window was resized. buffer size is {} x {}".format(width, height))

    def close(self):
        print("The window is closing")


def launch():
    sender, receiver = multiprocessing.Pipe()
    process = multiprocessing.Process(target=Test.run, args=())
    process.start()


def render(receiver):

    # main = threading.main_thread()
    # print(main)
    # asyncio.run_coroutine_threadsafe(mglw.run_window_config(Test), main)
    mglw.run_window_config(Test)
    sphere = mglw.geometry.sphere()
    print(f"Sphere: {sphere}")
    while True:
        if receiver.poll(.1):
            update = receiver.recv()
            print(f"Update to render: {update}")


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