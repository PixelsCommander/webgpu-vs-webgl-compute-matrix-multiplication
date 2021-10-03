async function initGPUDevice() {
    if (!navigator.gpu) {
        console.log(
            "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
        );
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.log("Failed to get GPU adapter.");
        return;
    }
    return await adapter.requestDevice();
}

async function multiplyWebGPU(matrixSize, device) {
    const firstMatrix = matrixRandomGPU(matrixSize);

    const gpuBufferFirstMatrix = device.createBuffer({
        mappedAtCreation: true,
        size: firstMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();

    // Second Matrix

    const secondMatrix = matrixRandomGPU(matrixSize);

    const startTime = Date.now();

    const gpuBufferSecondMatrix = device.createBuffer({
        mappedAtCreation: true,
        size: secondMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    // Result Matrix

    const resultMatrixBufferSize =
        Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
    const resultMatrixBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Bind group layout and bind group

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: gpuBufferFirstMatrix
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: gpuBufferSecondMatrix
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: resultMatrixBuffer
                }
            }
        ]
    });

    // Compute shader code

    const shaderModule = device.createShaderModule({
        code: `
      [[block]] struct Matrix {
        size : vec2<f32>;
        numbers: array<f32>;
      };
      
      [[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
      [[group(0), binding(1)]] var<storage, read> secondMatrix : Matrix;
      [[group(0), binding(2)]] var<storage, write> resultMatrix : Matrix;
      
      [[stage(compute), workgroup_size(8, 8)]]
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        // Guard against out-of-bounds work group sizes.
        if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(secondMatrix.size.y)) {
          return;
        }

        resultMatrix.size = vec2<f32>(firstMatrix.size.x, secondMatrix.size.y);
        
        let resultCell = vec2<u32>(global_id.x, global_id.y);
        var result = 0.0;
        for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
          let a = i + resultCell.x * u32(firstMatrix.size.y);
          let b = resultCell.y + i * u32(secondMatrix.size.y);
          result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
        }
        
        let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
        resultMatrix.numbers[index] = result;
      }
    `
    });

    // Pipeline setup

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: "main"
        }
    });

    // Commands submission

    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const x = Math.min(Math.ceil(firstMatrix[0] / 8), 256); // X dimension of the grid of workgroups to dispatch.
    const y = Math.min(Math.ceil(secondMatrix[1] / 8), 256); // Y dimension of the grid of workgroups to dispatch.
    passEncoder.dispatch(x, y);
    passEncoder.endPass();

    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
        resultMatrixBuffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        resultMatrixBufferSize /* size */
    );

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    window.webGPUarrayBuffer = gpuReadBuffer.getMappedRange();
    const endTime = Date.now();
    window.results[matrixSize] = window.results[matrixSize] || {};
    window.results[matrixSize]['WebGPU'] = endTime - startTime;
}

function matrixRandomGPU(n) {
    m = [n,n];
    for(var i=0;i<n*n;++i) {
        m.push(Math.random()*1);//000000000000);
    }
    return new Float32Array(m);
}