# yolo.py
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#


from __future__ import print_function

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import time


def _preprocess_yolo(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


class PostprocessYOLO(object):
    """Class for post-processing the three output tensors from YOLO."""

    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 nms_threshold,
                 yolo_input_resolution,
                 category_num=80):
        """Initialize with all values that will be kept when processing
        several frames.  Assuming 3 outputs of the network in the case
        of (large) YOLO, or 2 for the Tiny YOLO.
        Keyword arguments:
        yolo_masks -- a list of 3 (or 2) three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 (or 6) two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_wh -- tuple (W, H) for the target network
        category_num -- number of output categories/classes
        """
        self.masks = yolo_masks
        self.anchors = yolo_anchors
        self.nms_threshold = nms_threshold
        self.input_wh = (yolo_input_resolution[1], yolo_input_resolution[0])
        self.category_num = category_num

    def process(self, outputs, resolution_raw, conf_th):
        """Take the YOLO outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.
        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        conf_th -- confidence threshold, e.g. 0.3
        """
        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes_xywh, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw, conf_th)
        

        if len(boxes_xywh) > 0:
            # convert (x, y, width, height) to (x1, y1, x2, y2)
            img_w, img_h = resolution_raw
            xx = boxes_xywh[:, 0].reshape(-1, 1)
            yy = boxes_xywh[:, 1].reshape(-1, 1)
            ww = boxes_xywh[:, 2].reshape(-1, 1)
            hh = boxes_xywh[:, 3].reshape(-1, 1)
            boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., float(img_w-1))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., float(img_h-1))
            boxes = boxes.astype(np.int)
        else:
            boxes = np.zeros((0, 4), dtype=np.int)  # empty

        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.
        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + self.category_num)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, resolution_raw, conf_th):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.
        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        conf_th -- confidence threshold
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence, conf_th)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        width, height = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.
        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        def sigmoid_v(array):
            return np.reciprocal(np.exp(-array) + 1.0)

        def exponential_v(array):
            return np.exp(array)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., 0:2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4:5])
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_wh
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs, conf_th):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.
        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)
        conf_th -- confidence threshold
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= conf_th)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).
        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
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


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)
    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)
    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class YOLO_trt(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLOv3 or YOLOv4"""

    def _load_engine(self):
        TRTbin = './model_data/%s.trt' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()

    def _init_yolov3_postprocessor(self):
        h, w = self.input_shape
        filters = (self.category_num + 5) * 3
        if 'tiny' in self.model:
            self.output_shapes = [(1, filters, h // 32, w // 32),
                                  (1, filters, h // 16, w // 16)]
        else:
            self.output_shapes = [(1, filters, h // 32, w // 32),
                                  (1, filters, h // 16, w // 16),
                                  (1, filters, h //  8, w //  8)]
        if 'tiny' in self.model:
            postprocessor_args = {
                # A list of 2 three-dimensional tuples for the Tiny YOLO masks
                'yolo_masks': [(3, 4, 5), (0, 1, 2)],
                # A list of 6 two-dimensional tuples for the Tiny YOLO anchors
                'yolo_anchors': [(10, 14), (23, 27), (37, 58),
                                 (81, 82), (135, 169), (344, 319)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        else:
            postprocessor_args = {
                # A list of 3 three-dimensional tuples for the YOLO masks
                'yolo_masks': [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                # A list of 9 two-dimensional tuples for the YOLO anchors
#                 'yolo_anchors': [(10, 13), (16, 30), (33, 23),
#                                  (30, 61), (62, 45), (59, 119),
#                                  (116, 90), (156, 198), (373, 326)],
                'yolo_anchors': [(122, 78),  (144, 95),  (176, 117),  (209, 60),  (220, 153),  (315, 84),  (383, 291),  (577, 251),  (675, 387)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        self.postprocessor = PostprocessYOLO(**postprocessor_args)

    def _init_yolov4_postprocessor(self):
        h, w = self.input_shape
        filters = (self.category_num + 5) * 3
        if 'tiny' in self.model:
            self.output_shapes = [(1, filters, h // 16, w // 16),
                                  (1, filters, h // 32, w // 32)]
        else:
            self.output_shapes = [(1, filters, h //  8, w //  8),
                                  (1, filters, h // 16, w // 16),
                                  (1, filters, h // 32, w // 32)]
        if 'tiny' in self.model:
            postprocessor_args = {
                # A list of 2 three-dimensional tuples for the Tiny YOLO masks
                'yolo_masks': [(0, 1, 2), (3, 4, 5)],
                # A list of 6 two-dimensional tuples for the Tiny YOLO anchors
                'yolo_anchors': [(10, 14), (23, 27), (37, 58),
                                 (81, 82), (135, 169), (344, 319)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        else:
            postprocessor_args = {
                # A list of 3 three-dimensional tuples for the YOLO masks
                'yolo_masks': [(0, 1, 2), (3, 4, 5), (6, 7, 8)],
                # A list of 9 two-dimensional tuples for the YOLO anchors
                'yolo_anchors': [(12, 16), (19, 36), (40, 28),
                                 (36, 75), (76, 55), (72, 146),
                                 (142, 110), (192, 243), (459, 401)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        self.postprocessor = PostprocessYOLO(**postprocessor_args)

    def __init__(self, model, input_shape, category_num=80):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.model = model
        self.input_shape = input_shape
        self.category_num = category_num
        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2
        if 'yolov3' in self.model:
            self._init_yolov3_postprocessor()
        elif 'yolov4' in self.model:
            self._init_yolov4_postprocessor()
        else:
            raise ValueError('bad model name (%s)!' % args.model)
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self._create_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.outputs
        del self.inputs
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def detect(self, imgs, conf_th=0.3):
        """Detect objects in the input image."""
        shape_orig_WH = (imgs[0].shape[1], imgs[0].shape[0])
#         img_resized = _preprocess_yolo(img, self.input_shape)
#         s = time.time()
        img_resized = np.array([_preprocess_yolo(img, self.input_shape) for img in imgs])
#         print('Preprocess: ', time.time()-s)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)

        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

        # Before doing post-processing, we need to reshape the outputs
        # as do_inference() will give us flat arrays.
        

        trt_outputs = [
            output.reshape(shape)
            for output, shape in zip(trt_outputs, self.output_shapes)]

        # Run the post-processing algorithms on the TensorRT outputs
        # and get the bounding box details of detected objects
#         s2 = time.time()
        batch_boxes, batch_classes, batch_scores = [], [], []
        for batch_idx in range(trt_outputs[0].shape[0]):
            trt_output = [np.expand_dims(out[batch_idx], axis=0) for out in trt_outputs]
            boxes, classes, scores = self.postprocessor.process(
                trt_output, shape_orig_WH, conf_th)
            batch_boxes.append(boxes)
            batch_classes.append(classes)
            batch_scores.append(scores)
#         return boxes, scores, classes
#         print('Proprocess: ', time.time()-s2)
        return batch_boxes, batch_scores, batch_classes