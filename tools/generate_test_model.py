from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto


def _init_tensor(name: str, shape: list[int], rng: np.random.Generator) -> onnx.TensorProto:
    values = rng.standard_normal(np.prod(shape)).astype(np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, shape, values.flatten().tolist())


def build_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    hidden_dim2: int | None = None,
) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)

    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    w1 = _init_tensor("W1", [input_dim, hidden_dim], rng)
    b1 = _init_tensor("B1", [hidden_dim], rng)

    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["h1"]),
        helper.make_node("Add", ["h1", "B1"], ["h1b"]),
        helper.make_node("Relu", ["h1b"], ["h1r"]),
    ]

    initializers = [w1, b1]

    if hidden_dim2 is None:
        w2 = _init_tensor("W2", [hidden_dim, output_dim], rng)
        b2 = _init_tensor("B2", [output_dim], rng)
        nodes.extend(
            [
                helper.make_node("MatMul", ["h1r", "W2"], ["out"]),
                helper.make_node("Add", ["out", "B2"], ["output"]),
            ]
        )
        initializers.extend([w2, b2])
    else:
        w2 = _init_tensor("W2", [hidden_dim, hidden_dim2], rng)
        b2 = _init_tensor("B2", [hidden_dim2], rng)
        w3 = _init_tensor("W3", [hidden_dim2, output_dim], rng)
        b3 = _init_tensor("B3", [output_dim], rng)
        nodes.extend(
            [
                helper.make_node("MatMul", ["h1r", "W2"], ["h2"]),
                helper.make_node("Add", ["h2", "B2"], ["h2b"]),
                helper.make_node("Relu", ["h2b"], ["h2r"]),
                helper.make_node("MatMul", ["h2r", "W3"], ["out"]),
                helper.make_node("Add", ["out", "B3"], ["output"]),
            ]
        )
        initializers.extend([w2, b2, w3, b3])

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
    parser.add_argument("--out", required=True, help="Output ONNX file path")
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--hidden-dim2", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = build_model(args.input_dim, args.hidden_dim, args.output_dim, args.seed, args.hidden_dim2)
    onnx.save(model, args.out)


if __name__ == "__main__":
    main()
