import asyncio
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import tensorrt as trt
import numpy as np
import websockets
from websockets.server import serve
import msgpack
import msgpack_numpy
import time
import ctypes

# Patch msgpack
msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRTServer")

# --- CTYPES CUDART WRAPPER ---
libcudart = ctypes.CDLL("libcudart.so") # Use symlink

# Define constants
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

def check_cuda_err(err):
    if err != 0:
        raise RuntimeError(f"CUDA Error code: {err}")

def cudaMalloc(size):
    ptr = ctypes.c_void_p()
    err = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    check_cuda_err(err)
    return ptr.value

def cudaMemcpy(dst, src, count, kind):
    err = libcudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), count, kind)
    check_cuda_err(err)

def cudaStreamSynchronize(stream):
    err = libcudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    check_cuda_err(err)
    
def cudaFree(ptr):
    err = libcudart.cudaFree(ctypes.c_void_p(ptr))
    check_cuda_err(err)

# -----------------------------

class TensorRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            logger.info(f"Loading engine from {engine_path}...")
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.tensor_names = []
        
        # Allocate buffers
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_names.append(name)
            
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name).name # e.g. FLOAT
            
            # Size calc
            if -1 in shape:
                 min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, 0)
                 alloc_shape = max_shape
            else:
                 alloc_shape = shape
            
            np_dtype = None
            if dtype == 'FLOAT': np_dtype = np.float32
            elif dtype == 'HALF': np_dtype = np.float16
            elif dtype == 'INT32': np_dtype = np.int32
            elif dtype == 'INT8': np_dtype = np.int8
            elif dtype == 'BOOL': np_dtype = np.bool_
            
            size = np.dtype(np_dtype).itemsize * np.prod(alloc_shape)
            
            # Allocate device memory
            ptr = cudaMalloc(size)
                
            self.allocations.append(ptr)
            
            binding = {
                "name": name,
                "ptr": ptr,
                "shape": alloc_shape, 
                "dtype": np_dtype,
                "mode": mode,
                "size": size
            }
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
                
            # Set address
            self.context.set_tensor_address(name, ptr)

    def infer(self, input_dict):
        # input_dict: {name: np.array}
        
        # 1. Copy Inputs Host -> Device
        for inp in self.inputs:
            name = inp["name"]
            if name not in input_dict:
                continue
                
            data = input_dict[name]
            
            # Update input shape
            self.context.set_input_shape(name, data.shape)
            
            # Ensure contiguous and correct type
            data = np.ascontiguousarray(data, dtype=inp["dtype"])
            
            # Copy
            cudaMemcpy(inp["ptr"], data.ctypes.data, data.nbytes, cudaMemcpyHostToDevice)

        # 2. Execute
        self.context.execute_async_v3(0)
        
        # 3. Copy Outputs Device -> Host
        results = {}
        for out in self.outputs:
            name = out["name"]
            out_shape = self.context.get_tensor_shape(name)
            host_buffer = np.empty(out_shape, dtype=out["dtype"])
            
            cudaMemcpy(host_buffer.ctypes.data, out["ptr"], host_buffer.nbytes, cudaMemcpyDeviceToHost)
                 
            results[name] = host_buffer
            
        cudaStreamSynchronize(0)
        
        return results

model = None

async def handler(websocket):
    logger.info("Client connected")
    await websocket.send(msgpack.packb({"status": "ready"}))
    
    try:
        async for message in websocket:
            data = msgpack.unpackb(message)
            if model:
                outputs = model.infer(data)
                response = msgpack.packb(outputs)
                await websocket.send(response)
            else:
                await websocket.send(msgpack.packb({"error": "Model not loaded"}))
                
    except websockets.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error handling request: {e}")

async def main():
    global model
    import os
    engine_path = "model.trt" 
    # Use environment variable for flexibility
    onnx_path = os.environ.get("ONNX_TO_BUILD", "checkpoints/pi05_libero_pytorch/fp32/model.onnx")
    if "int8" in onnx_path:
        engine_path = "model_int8.trt"
    
    trt_path = engine_path
    
    import os
    if not os.path.exists(trt_path):
        logger.info("Building TRT engine...")
        # (Insert build logic here or just assume we run a builder script first)
        # Re-using verify_trt_python.py logic to build
        import trt_builder
        engine_bytes = trt_builder.build_engine(onnx_path)
        with open(trt_path, "wb") as f:
            f.write(engine_bytes)
    
    model = TensorRTModel(trt_path)
    
    async with serve(handler, "0.0.0.0", 8000, max_size=None):
        logger.info("Server started on port 8000")
        await asyncio.get_running_loop().create_future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
