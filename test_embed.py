import onnx
import os
onnx_path = "model/dinov2/dinov2.onnx"
model = onnx.load(onnx_path, load_external_data=True)
for tensor in model.graph.initializer:
    if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
        tensor.data_location = onnx.TensorProto.DEFAULT
        del tensor.external_data[:]
onnx.save(model, "test_dinov2.onnx")
print("Size:", os.path.getsize("test_dinov2.onnx"))
