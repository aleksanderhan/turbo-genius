import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import AutoTokenizer


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Example sentences
sentences = [
    "Tell me about the connection between the halting problem and the spectral gap problem.",
    "Tell me about the limits of computatability.",
    "Does TensorRT require representative data for calibration?",
    "What is the difference between Flash Attention and Flash Attention 2?",
    "Write me a python script to do supervised fine-tuning of a large language model!",
]

# Tokenize sentences and save as .npy files
for i, sentence in enumerate(sentences):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    np.save(f"calibration_data_{i}.npy", input_ids)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_files, batch_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.calibration_files = calibration_files
        self.current_index = 0

        # Load the first calibration file to determine input shape
        input_data = np.load(self.calibration_files[0])
        self.input_shape = input_data.shape[1:]  # Exclude batch size
        self.device_input = cuda.mem_alloc(trt.volume((self.batch_size, *self.input_shape)) * trt.float32.itemsize)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_files):
            return None

        batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        for i in range(self.batch_size):
            input_data = np.load(self.calibration_files[self.current_index + i])
            batch[i] = input_data

        self.current_index += self.batch_size
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
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

calibration_files = [f"calibration_data_{i}.npy" for i in range(len(sentences))]
calibrator = MyCalibrator(calibration_files, batch_size=1)

engine = build_engine("model/model.onnx", calibrator)


def infer(engine, input_data):
    context = engine.create_execution_context()
    input_shape = input_data.shape
    input_nbytes = input_data.nbytes
    output_shape = (input_shape[0], 512)  # Adjust output shape as needed
    output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data

input_data = tokenizer("This is a test input.", return_tensors="pt").input_ids.numpy()
output_data = infer(engine, input_data)
print(output_data)
