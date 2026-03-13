# input_neuron_bounds_ibp.py
ALGORITHM_NAME = "Input Neuron Bounds (Debug)"
IS_DETERMINISTIC = True

import numpy as np


def _get_initializer(graph, name: str):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None


def _read_float_data(initializer) -> np.ndarray:
    if len(initializer.float_data) > 0:
        return np.array(initializer.float_data, dtype=np.float32)
    elif len(initializer.raw_data) > 0:
        return np.frombuffer(initializer.raw_data, dtype=np.float32).copy()
    return np.zeros(int(np.prod(initializer.dims)), dtype=np.float32)


def _extract_directions(graph) -> np.ndarray:
    """
    Reads the actual directions from output_initializer_W.
    Shape is (n_features, n_directions). Each column is one direction vector
    restricted to the active (bridge) neuron rows.

    Returns: np.ndarray of shape (n_directions, 2) — one (a, b) per direction.
    """
    init = _get_initializer(graph, "output_initializer_W")
    if init is None:
        raise ValueError("Could not find 'output_initializer_W'.")

    n_features, n_directions = init.dims[0], init.dims[1]
    weights = _read_float_data(init).reshape(n_features, n_directions)

    # Active rows are the bridge neurons for the selected neuron pair
    active_rows = [i for i in range(n_features) if np.any(weights[i] != 0.0)]
    if len(active_rows) < 2:
        raise ValueError(f"Expected >=2 active rows, found {len(active_rows)}.")

    # Each column of the active submatrix is (d_a, d_b) for that direction
    # shape: (2, n_directions) -> transpose to (n_directions, 2)
    direction_matrix = weights[active_rows[:2], :]  # (2, N)
    directions = direction_matrix.T                  # (N, 2)

    print(f"[Input Neuron Bounds] Extracted directions shape: {directions.shape}")
    print(f"[Input Neuron Bounds] Direction 0:  {directions[0]}")
    print(f"[Input Neuron Bounds] Direction N/4: {directions[n_directions // 4]}")
    print(f"[Input Neuron Bounds] Direction N/2: {directions[n_directions // 2]}")

    return directions


def _extract_selected_neuron_indices(graph) -> list[int]:
    init = _get_initializer(graph, "output_initializer_W")
    n_features = init.dims[0]
    weights = _read_float_data(init).reshape(n_features, init.dims[1])
    return [i for i in range(n_features) if np.any(weights[i] != 0.0)]


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    Debug algorithm that reads both selected neurons AND directions directly
    from the modified model's weight matrix, then computes exact IBP bounds
    for the input box of those two neurons.

    The bounds are now guaranteed to match the directions used by compute_polygon.
    """
    graph = onnx_model.graph

    # ------------------------------------------------------------------ #
    # 1. Extract directions from weight matrix                            #
    # ------------------------------------------------------------------ #
    directions = _extract_directions(graph)      # shape (N, 2)
    n_directions = len(directions)

    # ------------------------------------------------------------------ #
    # 2. Extract selected neuron bounds                                   #
    # ------------------------------------------------------------------ #
    active_indices = _extract_selected_neuron_indices(graph)
    neuron_a_idx = active_indices[0]
    neuron_b_idx = active_indices[1]

    a_lo, a_hi = float(input_bounds[neuron_a_idx, 0]), float(input_bounds[neuron_a_idx, 1])
    b_lo, b_hi = float(input_bounds[neuron_b_idx, 0]), float(input_bounds[neuron_b_idx, 1])

    print(f"[Input Neuron Bounds] Neuron A (idx={neuron_a_idx}): [{a_lo}, {a_hi}]")
    print(f"[Input Neuron Bounds] Neuron B (idx={neuron_b_idx}): [{b_lo}, {b_hi}]")
    print(f"[Input Neuron Bounds] n_directions={n_directions}")

    # ------------------------------------------------------------------ #
    # 3. IBP projection using ACTUAL directions from weight matrix        #
    # ------------------------------------------------------------------ #
    cos_a = directions[:, 0]   # weight for neuron A per direction
    sin_b = directions[:, 1]   # weight for neuron B per direction

    lower = (np.where(cos_a >= 0, cos_a * a_lo, cos_a * a_hi)
           + np.where(sin_b >= 0, sin_b * b_lo, sin_b * b_hi))

    upper = (np.where(cos_a >= 0, cos_a * a_hi, cos_a * a_lo)
           + np.where(sin_b >= 0, sin_b * b_hi, sin_b * b_lo))

    for i, (lo, hi) in enumerate(zip(lower, upper)):
        a, b = directions[i]
        print(f"  direction {i:3d} (a={a:+.4f}, b={b:+.4f}): [{lo:+.6f}, {hi:+.6f}]")

    return np.stack([lower, upper], axis=1).astype(np.float32)