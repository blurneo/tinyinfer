import onnx
from onnx import numpy_helper

def onnx_datatype_to_npType(data_type):
    if data_type == 1:
        return np.float32
    elif data_type == 7:
        return np.int64
    else:
        raise TypeError("don't support data type")

def parser_initializer(initializer):
    name = initializer.name
    print(f"initializer name: {name}")
    dims = initializer.dims
    shape = [x for x in dims]
    print(f"initializer with shape:{shape}")
    dtype = initializer.data_type
    print(f"initializer with type: {onnx_datatype_to_npType(dtype)} ")
    # print tenth buffer
    weights = np.frombuffer(initializer.raw_data, dtype=onnx_datatype_to_npType(dtype))
    W= numpy_helper.to_array(initializer)
    print(f"initializer weights shape:", W.shape)

def parser_tensor(tensor, use='normal'):
    name = tensor.name
    print(f"{use} tensor name: {name}")
    data_type = tensor.type.tensor_type.elem_type
    print(f"{use} tensor data type: {data_type}")
    dims = tensor.type.tensor_type.shape.dim
    shape = []
    for i, dim in enumerate(dims):
        shape.append(dim.dim_value)
    print(f"{use} tensor with shape:{shape} ")

def parser_node(node):
    def attri_value(attri):
        if attri.type == 1:
            return attri.i
        elif attri.type == 7:
            return list(attri.ints)
    name = node.name
    print(f"node name:{name}")
    opType = node.op_type
    print(f"node op type:{opType}")
    inputs = list(node.input)
    print(f"node with {len(inputs)} inputs:{inputs}")
    outputs = list(node.output)
    print(f"node with {len(outputs)} outputs:{outputs}")
    attributes = node.attribute
    for attri in attributes:
        name = attri.name
        value = attri_value(attri)
        print(f"{name} with value:{value}")

def parser_info(onnx_model):
    ir_version = onnx_model.ir_version
    producer_name = onnx_model.producer_name
    producer_version = onnx_model.producer_version
    for info in [ir_version, producer_name, producer_version]:
        print("onnx model with info:{}".format(info))

def parser_inputs(onnx_graph):
    inputs = onnx_graph.input
    for input in inputs:
        parser_tensor(input, 'input')

def parser_outputs(onnx_graph):
    outputs = onnx_graph.output
    for output in outputs:
        parser_tensor(output, 'output')

def parser_graph_initializers(onnx_graph):
    initializers = onnx_graph.initializer
    for initializer in initializers:
        parser_initializer(initializer)

def parser_graph_nodes(onnx_graph):
    nodes = onnx_graph.node
    for node in nodes:
        parser_node(node)
        t = 1

def onnx_parser():
    model_path = 'models/mnist/mnist-8.onnx'
    model = onnx.load(model_path)
    # 0.
    parser_info(model)
    graph = model.graph
    # 1.
    parser_inputs(graph)
    # 2.
    parser_outputs(graph)
    # 3.
    parser_graph_initializers(graph)
    # 4.
    parser_graph_nodes(graph)

import onnx
import numpy as np

if __name__ == '__main__':
    onnx_parser()