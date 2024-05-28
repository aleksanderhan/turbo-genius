import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_files, batch_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.calibration_files = calibration_files
        self.current_index = 0
        self.device_input = cuda.mem_alloc(trt.volume((batch_size, *self.input_shape)) * trt.float32.itemsize)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_files):
            return None

        batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        for i in range(self.batch_size):
            input_data = self.load_data(self.calibration_files[self.current_index + i])
            batch[i] = input_data

        self.current_index += self.batch_size
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def load_data(self, file_path):
        # Implement your data loading logic here
        return np.random.randn(*self.input_shape).astype(np.float32)

    def read_calibration_cache(self):
        # If you have a cache, return it here
        return None

    def write_calibration_cache(self, cache):
        # Save the cache here if needed
        pass

def build_engine(onnx_file_path, calibrator):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30
        builder.int8_mode = True
        builder.int8_calibrator = calibrator
        
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        return builder.build_cuda_engine(network)

calibration_files = [f"calibration_data_{i}.npy" for i in range(5)]
calibrator = MyCalibrator(calibration_files, batch_size=1)

engine = build_engine("model.onnx", calibrator)
