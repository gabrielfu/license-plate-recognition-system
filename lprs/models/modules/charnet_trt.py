import numpy as np
import cv2
import logging

import tensorrt as trt
import pycuda.driver as cuda

import time

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
#    start = time.time()
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
#    print(f'do_inference time: {time.time()-start}')
    return [out.host for out in outputs]


class TrtCharNet(object):
    def __init__(self, engine_path, input_size, max_batch_size):
        cuda.init()
        self.TRT_LOGGER = trt.Logger(min_severity=trt.Logger.ERROR)
        self._init_trt(engine_path)
        self.input_size = input_size
        self.max_batch_size = max_batch_size

    def _get_engine(self, engine_path):
        # If a serialized engine exists, use it instead of building an engine.
        logging.info(f"Reading engine from file {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _init_trt(self, engine_path):
        self.engine = self._get_engine(engine_path)
        self.buffers = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.buffers

    def _preprocess_img(self, img):
        resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        return img_in

    def _preprocess_img_lst(self, img_lst):
#        start = time.time()
        imgs_array = np.array([self._preprocess_img(img) for img in img_lst])  # (b,c,h,w)
        imgs_array = np.ascontiguousarray(imgs_array)
#        print(f'yolo_trt _preprocess_img_lst [len{len(img_lst)}] time {time.time() - start}')
        return imgs_array

    def detect(self, img_lst):
        '''
        Allocate buffers each prediction on the fly to avoid context issues
        inputs:
            - img_lst: list of BGR np arrays
        outputs:
            - imgs_preds: list of list of preds (x1,y1,x2,y2,conf,cls_conf,cls)  [p.s. normalized coords]
        '''
        imgs_array = self._preprocess_img_lst(img_lst)

        self.inputs[0].host = imgs_array
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        trt_outputs[0] = trt_outputs[0].reshape(self.max_batch_size, 33)

        return trt_outputs
