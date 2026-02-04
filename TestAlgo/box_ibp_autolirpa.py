# box_ibp_autolirpa.py
ALGORITHM_NAME = "Box IBP (auto_LiRPA)"
IS_DETERMINISTIC = True

import numpy as np


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    Requires auto_LiRPA + torch + onnx2pytorch package.:
      pip install auto-LiRPA onnx2pytorch torch
    """
    try:
        import torch
        from onnx2pytorch import ConvertModel
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
    except Exception as e:
        raise ImportError("auto_LiRPA + torch + onnx2pytorch are required.") from e

    # ONNX -> PyTorch
    torch_model = ConvertModel(onnx_model).eval()
    lb = torch.tensor(input_bounds[:, 0], dtype=torch.float32).unsqueeze(0)
    ub = torch.tensor(input_bounds[:, 1], dtype=torch.float32).unsqueeze(0)
    x0 = (lb + ub) / 2.0

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    x = BoundedTensor(x0, ptb)

    bounded_model = BoundedModule(torch_model, (x0,))

    out_lb, out_ub = bounded_model.compute_bounds(x=(x,), method="IBP")
    out_lb = out_lb.reshape(-1).detach().cpu().numpy()
    out_ub = out_ub.reshape(-1).detach().cpu().numpy()
    return np.stack([out_lb, out_ub], axis=1)
