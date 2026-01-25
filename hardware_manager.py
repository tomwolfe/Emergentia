import torch
import numpy as np

class HardwareManager:
    """Manages hardware-specific logic, especially for MPS stability."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.is_mps = (self.device.type == 'mps')
        # OPTIMIZATION: Keep ODE integration on the same device as the model to avoid transfers
        self.mps_ode_on_cpu = False

    def get_ode_device(self):
        """Returns the device for ODE integration."""
        # OPTIMIZATION: Keep ODE integration on the same device as the model
        return self.device

    def to_ode_device(self, tensor):
        """Moves a tensor to the ODE integration device."""
        # OPTIMIZATION: No device transfer needed since ODE runs on same device
        return tensor

    def to_main_device(self, tensor):
        """Moves a tensor back to the main training device."""
        # OPTIMIZATION: No device transfer needed since ODE runs on same device
        return tensor

    def prepare_model_for_ode(self, model):
        """Ensures the model's ODE function is on the correct device."""
        # OPTIMIZATION: Keep ODE function on the same device as the model
        ode_device = self.device
        if hasattr(model, 'ode_func'):
            model.ode_func.to(ode_device)
        return model

    def handle_mps_stability(self, z):
        """Applies stability fixes for MPS if necessary."""
        if self.is_mps:
            # MPS often benefits from float32 and nan_to_num
            return torch.nan_to_num(z.to(torch.float32), nan=0.0, posinf=1e2, neginf=-1e2)
        return z
