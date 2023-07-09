from onnx import numpy_helper
import onnx

MODEL_PATH = "models/bvlcalexnet-9.onnx"
_model = onnx.load(MODEL_PATH)
INTIALIZERS=_model.graph.initializer
Weight=[]
# for layer in _model.layers:
#     weight = layer.weight
for initializer in INTIALIZERS:
    W= numpy_helper.to_array(initializer)
    Weight.append(W)
    print('layer name:',initializer.name)
    print('weight:', W.shape)