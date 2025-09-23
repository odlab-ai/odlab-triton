import os
import sys
import onnx
import jinja2

from onnx import TensorProto


ONNX_DTYPES_TO_TRITON = {
    'FLOAT': 'TYPE_FP32',
    'UINT8': 'TYPE_UINT8',
    'INT64': 'TYPE_INT64',
    'INT32': 'TYPE_INT32',
    'FLOAT16': 'TYPE_FP16',
    'DOUBLE': 'TYPE_FP64',
}


def sanitize_dims(dims):
    return [d if d > 0 else -1 for d in dims]

def render_config(model_name, io_info, max_batch_size=16):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
    template = env.get_template("config.pbtxt.j2")

    return template.render(
        model_name=model_name,
        max_batch_size=max_batch_size,
        inputs=io_info['inputs'],
        outputs=io_info['outputs']
    )

def extract_io_info(onnx_path):
    model = onnx.load(onnx_path)

    def get_tensor_info(tensor):
        name = tensor.name
        shape = sanitize_dims([dim.dim_value for dim in tensor.type.tensor_type.shape.dim[1:]])
        elem_type = tensor.type.tensor_type.elem_type
        dtype = ONNX_DTYPES_TO_TRITON[TensorProto.DataType.Name(elem_type)]
        return {'name': name, 'dims': shape, 'data_type': dtype}

    inputs = [get_tensor_info(t) for t in model.graph.input]
    outputs = [get_tensor_info(t) for t in model.graph.output]

    return {'inputs': inputs, 'outputs': outputs}

def render_config(model_name, io_info, max_batch_size=16):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
    template = env.get_template("config.pbtxt.j2")

    return template.render(
        model_name=model_name,
        max_batch_size=max_batch_size,
        inputs=io_info['inputs'],
        outputs=io_info['outputs']
    )

def generate_config(onnx_path, save_dir):
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    io_info = extract_io_info(onnx_path)
    config_text = render_config(name, io_info)
    with open(os.path.join(save_dir, "config.pbtxt"), "w") as f:
        f.write(config_text)

    print(f"✅ Created config.pbtxt for {name}")



if __name__ == "__main__":
    onnx_path = sys.argv[1]
    output_dir = sys.argv[2]
    generate_config(onnx_path, output_dir)