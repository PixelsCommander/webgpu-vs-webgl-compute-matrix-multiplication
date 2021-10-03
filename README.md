# webgl-webgpu-matrix-benchmark

## High Performance Matrix Multiplication Utilizing WebGL and WebGPU.

WebGPU compute shaders performance is expected to be superior to the one you can get with WebGL.
This should happen because WebGPU does not have overhead of creating / initializing canvas and writing/reading from framebuffers.
The benchmark is built in order to experimentally confirm this assumption and also determine by how much WebGPU performance is superior to WebGL.