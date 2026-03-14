from __future__ import annotations

from pathlib import Path

import onnx

from generate_test_model import build_model


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "TestFiles" / "simple_4_hidden_layer_net x20000.onnx"
INPUT_DIM = 4
HIDDEN_DIMS = [5000, 5000, 5000, 5000]
OUTPUT_DIM = 2
SEED = 42


def main() -> None:
    model = build_model(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        seed=SEED,
    )
    onnx.save(model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
