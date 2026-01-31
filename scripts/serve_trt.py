import asyncio
import logging
import sys
import os

# [Jetson Fix] Inject system path for TensorRT if not found in venv
try:
    import tensorrt as trt
except ImportError:
    sys.path.append("/usr/lib/python3.12/dist-packages")
    import tensorrt as trt

sys.path.append(os.path.join(os.path.dirname(__file__)))
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

# Calibration Data Collection
CALIBRATION_FILE = "calibration_data.pt"
COLLECT_CALIBRATION = False
CALIBRATION_SAMPLES = 100
_calibration_buffer = []
# import torch



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

def cudaStreamCreate():
    ptr = ctypes.c_void_p()
    err = libcudart.cudaStreamCreate(ctypes.byref(ptr))
    check_cuda_err(err)
    return ptr.value

def cudaStreamDestroy(stream):
    err = libcudart.cudaStreamDestroy(ctypes.c_void_p(stream))
    check_cuda_err(err)

def cudaMemcpyAsync(dst, src, count, kind, stream):
    err = libcudart.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), count, kind, ctypes.c_void_p(stream))
    check_cuda_err(err)



def cudaStreamBeginCapture(stream, mode=0):
    err = libcudart.cudaStreamBeginCapture(ctypes.c_void_p(stream), mode)
    check_cuda_err(err)

def cudaStreamEndCapture(stream, pGraph):
    err = libcudart.cudaStreamEndCapture(ctypes.c_void_p(stream), pGraph)
    check_cuda_err(err)

def cudaGraphInstantiate(pGraphExec, graph, flags=0):
    # cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags, char *logBuffer, size_t bufferSize);
    # Using older signature or newer depending on CUDA version. 
    # Let's try the simple one or check libcudart symbols?
    # For simplified python access, we might need to handle the pointer complexity.
    # Assuming CUDA 11+ signature:
    # cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);
    # Actually, let's use the valid signature.
    err = libcudart.cudaGraphInstantiate(pGraphExec, graph, ctypes.c_void_p(0), ctypes.c_void_p(0), 0)
    check_cuda_err(err)

def cudaGraphLaunch(graphExec, stream):
    err = libcudart.cudaGraphLaunch(graphExec, ctypes.c_void_p(stream))
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

        self.graph_exec = None
        self.graph = None
        self.inference_count = 0
        self.stream = cudaStreamCreate()


    def infer(self, input_dict):
        # input_dict: {name: np.array}
        
        # Mapping from Engine Input Names (Long) to Client Protocol Names (Short)
        # Protocol defined in TensorRTRemotePolicy.extract_inputs
        KEY_MAPPING = {
            "observation.images.base_0_rgb": "base_0_rgb",
            "observation.images.left_wrist_0_rgb": "left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb": "right_wrist_0_rgb",
            "observation.state": "state",
            "observation.tokenized_prompt": "tokenized_prompt",
            "observation.tokenized_prompt_mask": "tokenized_prompt_mask",
            "noise": "noise",
            "prompt": "tokenized_prompt",
            "prompt_mask": "tokenized_prompt_mask"
        }

        # 1. Copy Inputs Host -> Device
        for inp in self.inputs:
            name = inp["name"]
            
            # Resolve data key
            data = None
            if name in input_dict:
                data = input_dict[name]
            elif name in KEY_MAPPING and KEY_MAPPING[name] in input_dict:
                data = input_dict[KEY_MAPPING[name]]
            

            
            if data is None:
                logger.warning(f"Warning: Input '{name}' not found in request! Available keys: {list(input_dict.keys())}")
                continue
                
            # Update input shape
            self.context.set_input_shape(name, data.shape)
            
            # Ensure contiguous and correct type
            data = np.ascontiguousarray(data, dtype=inp["dtype"])
            
            # [DEBUG] Check for NaNs/Infs
            if np.issubdtype(data.dtype, np.floating):
                if np.isnan(data).any() or np.isinf(data).any():
                    logger.error(f"ERROR: Input '{name}' contains NaNs or Infs!")
                    logger.error(f"  Min: {np.min(data)}, Max: {np.max(data)}")
                else:
                    logger.info(f"DEBUG: Input '{name}' OK. Range: [{np.min(data)}, {np.max(data)}]")
            else:
                 logger.info(f"DEBUG: Input '{name}' OK (Integer). Shape: {data.shape}")
            
            # Copy Async
            cudaMemcpyAsync(inp["ptr"], data.ctypes.data, data.nbytes, cudaMemcpyHostToDevice, self.stream)

        # 2. Execute
        
        # --- Save Calibration Data ---
        global _calibration_buffer, COLLECT_CALIBRATION
        if COLLECT_CALIBRATION and len(_calibration_buffer) < CALIBRATION_SAMPLES:
            # Reconstruct the full input dictionary that matches the model's expected arguments
            # We need to map back from engine names to what the model wrapper expects
            # Actually, we can just save the 'input_dict' passed to infer(), as that contains
            # 'tokenized_prompt', 'base_0_rgb', etc. which matches our export script's forward signature inputs.
            # We just need to ensure we capture numpy arrays.
            
            # We want to save a tuple/dict that can be unpacked into the model's forward()
            # OnnxWrapperModelOpt.forward(self, base_rgb, left_rgb, right_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise)
            
            # Input dict has: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise
            try:
                sample = (
                    input_dict["base_0_rgb"],
                    input_dict["left_wrist_0_rgb"],
                    input_dict["right_wrist_0_rgb"],
                    input_dict["state"],
                    input_dict["tokenized_prompt"],
                    input_dict["tokenized_prompt_mask"],
                    input_dict["noise"]
                )
                _calibration_buffer.append(sample)
                logger.info(f"Captured calibration sample {len(_calibration_buffer)}/{CALIBRATION_SAMPLES}")
                
                if len(_calibration_buffer) >= CALIBRATION_SAMPLES:
                     logger.info(f"Saving {len(_calibration_buffer)} calibration samples to {CALIBRATION_FILE}...")
                     import pickle
                     with open(CALIBRATION_FILE, 'wb') as f:
                        pickle.dump(_calibration_buffer, f)
                     # torch.save(_calibration_buffer, CALIBRATION_FILE)
                     COLLECT_CALIBRATION = False
                     logger.info("Calibration data saved.")
            except KeyError as e:
                logger.warning(f"Skipping calibration capture due to missing key: {e}")
        # -----------------------------

        # 2. Execute
        if self.graph_exec:
            cudaGraphLaunch(self.graph_exec, self.stream)
        else:
            if self.inference_count == 10:
                logger.info("Starting CUDA Graph Capture...")
                try:
                    cudaStreamBeginCapture(self.stream, 0) # global capture
                    self.context.execute_async_v3(self.stream)
                    
                    self.graph = ctypes.c_void_p()
                    cudaStreamEndCapture(self.stream, ctypes.byref(self.graph))
                    
                    self.graph_exec = ctypes.c_void_p()
                    cudaGraphInstantiate(ctypes.byref(self.graph_exec), self.graph, 0)
                    
                    logger.info("CUDA Graph Captured and Instantiated!")
                    cudaGraphLaunch(self.graph_exec, self.stream)
                except Exception as e:
                    logger.error(f"Graph capture failed: {e}. Falling back to standard execution.")
                    self.graph_exec = None
                    self.context.execute_async_v3(self.stream)
            else:
                self.context.execute_async_v3(self.stream)
        
        self.inference_count += 1
        
        # 3. Copy Outputs Device -> Host
        # Engine output name: "actions". Client expects "actions".
        results = {}
        for out in self.outputs:
            name = out["name"]
            out_shape = self.context.get_tensor_shape(name)
            host_buffer = np.empty(out_shape, dtype=out["dtype"])
            
            cudaMemcpyAsync(host_buffer.ctypes.data, out["ptr"], host_buffer.nbytes, cudaMemcpyDeviceToHost, self.stream)
                  
            results[name] = host_buffer
            
        cudaStreamSynchronize(self.stream)
        
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine_path = args.engine_path
    port = args.port
    
    if not os.path.exists(engine_path):
        logger.error(f"Engine not found at {engine_path}")
        return

    model = TensorRTModel(engine_path)
    
    async with serve(handler, "0.0.0.0", port, max_size=None):
        logger.info(f"Server started on port {port}")
        await asyncio.get_running_loop().create_future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
