from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper


def _init_tensor(name: str, shape: list[int], rng: np.random.Generator) -> onnx.TensorProto:
    """
    Initialize a tensor from the given shape.
    """
    values = rng.standard_normal(np.prod(shape)).astype(np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, shape, values.flatten().tolist())


def build_model(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    seed: int,
) -> onnx.ModelProto:
    """
    Build a simple ONNX model.
    """
    rng = np.random.default_rng(seed)

    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    if not hidden_dims:
        raise ValueError("At least one hidden layer is required.")

    nodes = []
    initializers = []
    previous_output = "input"
    previous_dim = input_dim

    for layer_index, hidden_dim in enumerate(hidden_dims, start=1):
        weight_name = f"W{layer_index}"
        bias_name = f"B{layer_index}"
        linear_output = f"h{layer_index}b"
        activation_output = f"h{layer_index}r"
        initializers.extend(
            [
                _init_tensor(weight_name, [previous_dim, hidden_dim], rng),
                _init_tensor(bias_name, [hidden_dim], rng),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Gemm",
                    [previous_output, weight_name, bias_name],
                    [linear_output],
                    name=f"gemm_{layer_index}",
                ),
                helper.make_node(
                    "Relu",
                    [linear_output],
                    [activation_output],
                    name=f"relu_{layer_index}",
                ),
            ]
        )
        previous_output = activation_output
        previous_dim = hidden_dim

    output_layer_index = len(hidden_dims) + 1
    initializers.extend(
        [
            _init_tensor(f"W{output_layer_index}", [previous_dim, output_dim], rng),
            _init_tensor(f"B{output_layer_index}", [output_dim], rng),
        ]
    )
    nodes.append(
        helper.make_node(
            "Gemm",
            [previous_output, f"W{output_layer_index}", f"B{output_layer_index}"],
            ["output"],
            name=f"gemm_{output_layer_index}",
        )
    )

    graph = helper.make_graph(
        nodes,
        "simple_3_layer_net",
        [input_value],
        [output_value],
        initializer=initializers,
    )

    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 11
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a minimal test ONNX model.")
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--hidden-dim2", type=int, default=None)
    parser.add_argument(
        "--hidden-dims",
        help="Comma-separated hidden layer sizes. Overrides --hidden-dim/--hidden-dim2 when provided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hidden_dims:
        hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
    else:
        hidden_dims = [args.hidden_dim]
        if args.hidden_dim2 is not None:
            hidden_dims.append(args.hidden_dim2)

    model = build_model(args.input_dim, hidden_dims, args.output_dim, args.seed)
    onnx.save_model(model, "Test8", "textproto", save_as_external_data=True)


if __name__ == "__main__":
    main()
