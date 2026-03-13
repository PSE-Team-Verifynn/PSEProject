# direction_count_ibp.py
ALGORITHM_NAME = "Box IBP (Debug)"
IS_DETERMINISTIC = True

import numpy as np
import onnx
from onnx import numpy_helper


def _get_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy Interval Bound Propagation over an ONNX graph.

    Supported ops: Constant, ConstantOfShape, Gemm, MatMul, Add, Sub, Mul, Div,
                   Relu, LeakyRelu, Sigmoid, Tanh, Flatten, Reshape, Transpose,
                   Concat, Squeeze, Unsqueeze, Shape, Gather, Cast, Slice, Pad,
                   BatchNormalization.
    """
    weights = _get_initializers(onnx_model)
    lb_map: dict = {}
    ub_map: dict = {}

    # ── FIX 1: find real data input ──────────────────────────────────────────
    # Old ONNX opset (< 9) lists every initializer as a graph.input, so
    # graph.input[0] may point at a weight tensor rather than the data feed.
    # Filter them out to find the genuine data input.
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    data_inputs = [inp for inp in onnx_model.graph.input
                   if inp.name not in initializer_names]
    if not data_inputs:
        raise ValueError(
            "No non-initializer input found in the ONNX graph. "
            "Inputs seen: " + str([i.name for i in onnx_model.graph.input])
        )
    input_name = data_inputs[0].name
    lb_map[input_name] = input_bounds[:, 0].astype(np.float64)
    ub_map[input_name] = input_bounds[:, 1].astype(np.float64)

    # ── Accessors (weights are point intervals) ───────────────────────────────
    def get_lb(name: str) -> np.ndarray:
        if name in lb_map:
            return lb_map[name]
        return weights[name].astype(np.float64)

    def get_ub(name: str) -> np.ndarray:
        if name in ub_map:
            return ub_map[name]
        return weights[name].astype(np.float64)

    # ── Core IBP: bounds for  W @ x + b,  W must be (out, in) ────────────────
    def affine_ibp(W: np.ndarray,
                   lb_in: np.ndarray,
                   ub_in: np.ndarray,
                   bias: np.ndarray | None = None):
        lb_in = lb_in.flatten()
        ub_in = ub_in.flatten()
        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)
        lb_out = W_pos @ lb_in + W_neg @ ub_in
        ub_out = W_pos @ ub_in + W_neg @ lb_in
        if bias is not None:
            lb_out += bias.flatten()
            ub_out += bias.flatten()
        return lb_out, ub_out

    # ── Forward pass ─────────────────────────────────────────────────────────
    for node in onnx_model.graph.node:
        op      = node.op_type
        inputs  = list(node.input)
        outputs = list(node.output)

        # ── FIX 2: Constant ───────────────────────────────────────────────
        # Constant nodes produce tensors that don't live in the initializer
        # dict (weights).  Without this case, any downstream node that reads
        # a Constant output raises "No bounds found for: /Constant_output_0".
        if op == "Constant":
            val = next((a for a in node.attribute if a.name == "value"), None)
            if val is not None:
                arr = numpy_helper.to_array(val.t).astype(np.float64)
            else:
                fv  = next((a.f for a in node.attribute if a.name == "value_float"), None)
                iv  = next((a.i for a in node.attribute if a.name == "value_int"),   None)
                arr = np.array([fv if fv is not None else (iv if iv is not None else 0)],
                               dtype=np.float64)
            lb_map[outputs[0]] = ub_map[outputs[0]] = arr
            continue

        elif op == "ConstantOfShape":
            shape = get_lb(inputs[0]).astype(int).flatten().tolist()
            val   = next((a for a in node.attribute if a.name == "value"), None)
            fill  = float(numpy_helper.to_array(val.t).flat[0]) if val else 0.0
            arr   = np.full(shape, fill, dtype=np.float64)
            lb_map[outputs[0]] = ub_map[outputs[0]] = arr
            continue

        # ── FIX 3: Gemm – transB logic was inverted ───────────────────────
        # ONNX stores FC weights as (out, in).
        # transB=1 (PyTorch default): op computes A @ B^T → B is already
        #   (out, in), which is exactly what affine_ibp expects → keep B.
        # transB=0 (rare):            op computes A @ B   → B is (in, out),
        #   must transpose to (out, in) before calling affine_ibp.
        elif op == "Gemm":
            alpha  = next((a.f for a in node.attribute if a.name == "alpha"),  1.0)
            beta   = next((a.f for a in node.attribute if a.name == "beta"),   1.0)
            transB = next((a.i for a in node.attribute if a.name == "transB"), 0)

            lb_a = get_lb(inputs[0]).flatten()
            ub_a = get_ub(inputs[0]).flatten()
            W    = weights[inputs[1]].astype(np.float64)
            bias = (weights[inputs[2]].astype(np.float64).flatten() * beta
                    if len(inputs) > 2 and inputs[2] else None)

            # Normalize W → (out, in)
            if not transB:         # stored (in, out) → flip
                W = W.T
            # else: stored (out, in) → already correct
            W = W * alpha

            lb_out, ub_out = affine_ibp(W, lb_a, ub_a, bias)

        # ── FIX 4: MatMul – handle x@W and W@x; both-dynamic fallback ────
        # Standard ONNX MatMul: Y = A @ B.
        #   • B is weight (in, out): W_eff = B.T → (out, in), data = A
        #   • A is weight (out, in): W_eff = A,              data = B
        #   • Both dynamic: use midpoint of B as a proxy weight
        elif op == "MatMul":
            b_is_weight = inputs[1] in weights
            a_is_weight = inputs[0] in weights

            if b_is_weight:
                W    = weights[inputs[1]].astype(np.float64)
                lb_a = get_lb(inputs[0]).flatten()
                ub_a = get_ub(inputs[0]).flatten()
                W_eff = W.T if W.ndim >= 2 else W.reshape(1, -1)   # (out, in)
                lb_out, ub_out = affine_ibp(W_eff, lb_a, ub_a)

            elif a_is_weight:
                W    = weights[inputs[0]].astype(np.float64)
                lb_b = get_lb(inputs[1]).flatten()
                ub_b = get_ub(inputs[1]).flatten()
                W_eff = W if W.ndim >= 2 else W.reshape(-1, 1)      # (out, in)
                lb_out, ub_out = affine_ibp(W_eff, lb_b, ub_b)

            else:
                # Both dynamic (e.g. attention layers): treat midpoint of B as W.
                # This gives valid (if imprecise) bounds without crashing.
                lb_a  = get_lb(inputs[0]).flatten()
                ub_a  = get_ub(inputs[0]).flatten()
                lb_b  = get_lb(inputs[1])
                ub_b  = get_ub(inputs[1])
                W_mid = (lb_b + ub_b) / 2.0
                W_eff = W_mid.T if W_mid.ndim >= 2 else W_mid.reshape(1, -1)
                lb_out, ub_out = affine_ibp(W_eff, lb_a, ub_a)

        # ── Add ───────────────────────────────────────────────────────────
        elif op == "Add":
            lb_a, ub_a = get_lb(inputs[0]), get_ub(inputs[0])
            lb_b, ub_b = get_lb(inputs[1]), get_ub(inputs[1])
            lb_out = lb_a + lb_b
            ub_out = ub_a + ub_b

        # ── Sub ───────────────────────────────────────────────────────────
        elif op == "Sub":
            lb_a, ub_a = get_lb(inputs[0]), get_ub(inputs[0])
            lb_b, ub_b = get_lb(inputs[1]), get_ub(inputs[1])
            lb_out = lb_a - ub_b
            ub_out = ub_a - lb_b

        # ── Mul ───────────────────────────────────────────────────────────
        elif op == "Mul":
            lb_a, ub_a = get_lb(inputs[0]), get_ub(inputs[0])
            lb_b, ub_b = get_lb(inputs[1]), get_ub(inputs[1])
            c      = np.stack([lb_a * lb_b, lb_a * ub_b, ub_a * lb_b, ub_a * ub_b])
            lb_out = c.min(axis=0)
            ub_out = c.max(axis=0)

        # ── Div ───────────────────────────────────────────────────────────
        elif op == "Div":
            lb_a, ub_a = get_lb(inputs[0]), get_ub(inputs[0])
            lb_b, ub_b = get_lb(inputs[1]), get_ub(inputs[1])
            c      = np.stack([lb_a / lb_b, lb_a / ub_b, ub_a / lb_b, ub_a / ub_b])
            lb_out = c.min(axis=0)
            ub_out = c.max(axis=0)

        # ── Relu ──────────────────────────────────────────────────────────
        elif op == "Relu":
            lb_x, ub_x = get_lb(inputs[0]), get_ub(inputs[0])
            lb_out = np.maximum(0.0, lb_x)
            ub_out = np.maximum(0.0, ub_x)

        # ── LeakyRelu ─────────────────────────────────────────────────────
        elif op == "LeakyRelu":
            alpha  = next((a.f for a in node.attribute if a.name == "alpha"), 0.01)
            lb_x, ub_x = get_lb(inputs[0]), get_ub(inputs[0])
            c      = np.stack([lb_x, ub_x, alpha * lb_x, alpha * ub_x])
            lb_out = c.min(axis=0)
            ub_out = c.max(axis=0)

        # ── Sigmoid (monotone ↑) ──────────────────────────────────────────
        elif op == "Sigmoid":
            lb_x, ub_x = get_lb(inputs[0]), get_ub(inputs[0])
            lb_out = 1.0 / (1.0 + np.exp(-lb_x))
            ub_out = 1.0 / (1.0 + np.exp(-ub_x))

        # ── Tanh (monotone ↑) ─────────────────────────────────────────────
        elif op == "Tanh":
            lb_x, ub_x = get_lb(inputs[0]), get_ub(inputs[0])
            lb_out = np.tanh(lb_x)
            ub_out = np.tanh(ub_x)

        # ── Flatten ───────────────────────────────────────────────────────
        elif op == "Flatten":
            axis  = next((a.i for a in node.attribute if a.name == "axis"), 1)
            lb_in = get_lb(inputs[0])
            ub_in = get_ub(inputs[0])
            if lb_in.ndim <= 1:
                lb_out, ub_out = lb_in, ub_in
            else:
                flat   = int(np.prod(lb_in.shape[axis:]))
                pre    = lb_in.shape[:axis]
                lb_out = lb_in.reshape(pre + (flat,))
                ub_out = ub_in.reshape(pre + (flat,))

        # ── FIX 5: Reshape – use the actual shape tensor ──────────────────
        # Previously this just called .flatten(), which ignored the target
        # shape entirely and broke any Reshape that changes rank (e.g. ViT's
        # patch embeddings: 50 tokens with 768 dims → (50, 768)).
        elif op == "Reshape":
            lb_in = get_lb(inputs[0])
            ub_in = get_ub(inputs[0])
            if len(inputs) > 1 and inputs[1]:
                raw = get_lb(inputs[1]).astype(int).flatten().tolist()
                # 0 → keep original dim at that position; -1 → infer
                orig  = list(lb_in.shape)
                shape = [orig[i] if s == 0 and i < len(orig) else s
                         for i, s in enumerate(raw)]
            else:
                shape = list(next(
                    (a.ints for a in node.attribute if a.name == "shape"), [-1]))
            try:
                lb_out = lb_in.reshape(shape)
                ub_out = ub_in.reshape(shape)
            except ValueError:
                # Shape tensor and actual size disagree (e.g. batch dim present
                # in graph but absent from our flat representation) → flatten.
                lb_out = lb_in.flatten()
                ub_out = ub_in.flatten()

        # ── Transpose ─────────────────────────────────────────────────────
        elif op == "Transpose":
            perm   = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
            lb_out = np.transpose(get_lb(inputs[0]), perm)
            ub_out = np.transpose(get_ub(inputs[0]), perm)

        # ── FIX 6: Concat ─────────────────────────────────────────────────
        # dist_shift/mnist_concat concatenates inputs + cached features.
        # Without this, the Add node downstream gets mismatched shapes.
        elif op == "Concat":
            axis   = next((a.i for a in node.attribute if a.name == "axis"), 0)
            lbs    = [get_lb(i) for i in inputs if i]
            ubs    = [get_ub(i) for i in inputs if i]
            lb_out = np.concatenate(lbs, axis=axis)
            ub_out = np.concatenate(ubs, axis=axis)

        # ── Squeeze ───────────────────────────────────────────────────────
        elif op == "Squeeze":
            lb_in = get_lb(inputs[0])
            ub_in = get_ub(inputs[0])
            if len(inputs) > 1 and inputs[1]:
                axes = tuple(get_lb(inputs[1]).astype(int).flatten().tolist())
            else:
                al   = [list(a.ints) for a in node.attribute if a.name == "axes"]
                axes = tuple(al[0]) if al else None
            lb_out = np.squeeze(lb_in, axis=axes)
            ub_out = np.squeeze(ub_in, axis=axes)

        # ── Unsqueeze ─────────────────────────────────────────────────────
        elif op == "Unsqueeze":
            lb_in = get_lb(inputs[0])
            ub_in = get_ub(inputs[0])
            if len(inputs) > 1 and inputs[1]:
                axes = sorted(get_lb(inputs[1]).astype(int).flatten().tolist())
            else:
                al   = [list(a.ints) for a in node.attribute if a.name == "axes"]
                axes = sorted(al[0]) if al else []
            for ax in axes:
                lb_in = np.expand_dims(lb_in, axis=ax)
                ub_in = np.expand_dims(ub_in, axis=ax)
            lb_out, ub_out = lb_in, ub_in

        # ── Shape ─────────────────────────────────────────────────────────
        # Returns the shape of a tensor as a 1-D integer tensor; needed so
        # downstream Reshape nodes can read the correct target shape.
        elif op == "Shape":
            arr = np.array(get_lb(inputs[0]).shape, dtype=np.float64)
            lb_map[outputs[0]] = ub_map[outputs[0]] = arr
            continue

        # ── Gather ────────────────────────────────────────────────────────
        elif op == "Gather":
            axis    = next((a.i for a in node.attribute if a.name == "axis"), 0)
            data_lb = get_lb(inputs[0])
            data_ub = get_ub(inputs[0])
            idx     = get_lb(inputs[1]).astype(int)
            lb_out  = np.take(data_lb, idx, axis=axis)
            ub_out  = np.take(data_ub, idx, axis=axis)

        # ── Cast ──────────────────────────────────────────────────────────
        elif op == "Cast":
            lb_out = get_lb(inputs[0]).astype(np.float64)
            ub_out = get_ub(inputs[0]).astype(np.float64)

        # ── Slice ─────────────────────────────────────────────────────────
        elif op == "Slice":
            data_lb = get_lb(inputs[0])
            data_ub = get_ub(inputs[0])
            starts  = get_lb(inputs[1]).astype(int).flatten()
            ends    = get_lb(inputs[2]).astype(int).flatten()
            ax_inp  = (get_lb(inputs[3]).astype(int).flatten()
                       if len(inputs) > 3 and inputs[3] else range(len(starts)))
            steps   = (get_lb(inputs[4]).astype(int).flatten()
                       if len(inputs) > 4 and inputs[4] else [1] * len(starts))
            idx = [slice(None)] * data_lb.ndim
            for ax, st, en, sp in zip(ax_inp, starts, ends, steps):
                idx[int(ax)] = slice(int(st), int(en), int(sp))
            lb_out = data_lb[tuple(idx)]
            ub_out = data_ub[tuple(idx)]

        # ── Pad ───────────────────────────────────────────────────────────
        elif op == "Pad":
            lb_in = get_lb(inputs[0])
            ub_in = get_ub(inputs[0])
            if len(inputs) > 1 and inputs[1]:
                pads = get_lb(inputs[1]).astype(int).flatten().tolist()
            else:
                pads = list(next((a.ints for a in node.attribute if a.name == "pads"), []))
            const_val = (float(get_lb(inputs[2]).flat[0])
                         if len(inputs) > 2 and inputs[2] else 0.0)
            n         = lb_in.ndim
            pad_width = [(pads[i], pads[i + n]) for i in range(n)]
            lb_out = np.pad(lb_in, pad_width, constant_values=const_val)
            ub_out = np.pad(ub_in, pad_width, constant_values=const_val)

        # ── BatchNormalization ────────────────────────────────────────────
        elif op == "BatchNormalization":
            lb_x     = get_lb(inputs[0])
            ub_x     = get_ub(inputs[0])
            gamma    = weights[inputs[1]].astype(np.float64)
            beta_bn  = weights[inputs[2]].astype(np.float64)
            mean     = weights[inputs[3]].astype(np.float64)
            variance = weights[inputs[4]].astype(np.float64)
            eps      = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
            scale    = gamma / np.sqrt(variance + eps)
            shift    = beta_bn - scale * mean
            if lb_x.ndim > 1:               # broadcast over spatial dims
                view  = (-1,) + (1,) * (lb_x.ndim - 1)
                scale = scale.reshape(view)
                shift = shift.reshape(view)
            lb_s   = scale * lb_x + shift
            ub_s   = scale * ux_x + shift   # note: flip bounds where scale < 0
            lb_out = np.minimum(lb_s, ub_s)
            ub_out = np.maximum(lb_s, ub_s)

        else:
            raise NotImplementedError(
                f"IBP: unsupported ONNX op '{op}'. "
                f"Add a handler for it in direction_count_ibp.py."
            )

        lb_map[outputs[0]] = lb_out
        ub_map[outputs[0]] = ub_out

    # ── Collect output bounds ─────────────────────────────────────────────────
    output_name  = onnx_model.graph.output[0].name
    lower_bounds = lb_map[output_name].flatten()
    upper_bounds = ub_map[output_name].flatten()
    return np.stack([lower_bounds, upper_bounds], axis=1)