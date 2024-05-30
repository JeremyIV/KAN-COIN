from copy import deepcopy
import torch
import numpy as np

def quantize_and_dequantize(model, bits=9):
    
    q_model = deepcopy(model)
    # Calculate the number of quantization levels
    num_levels = 2 ** bits - 1

    # Iterate through all parameters in the model
    for name, param in q_model.named_parameters():
        with torch.no_grad():  # Ensure no gradients are computed in this operation
            # Store original data
            original_param = param.data.clone()

            # Find the scale and zero_point for quantization
            original_param_np = original_param.detach().cpu().numpy()
            min_val = np.percentile(original_param_np, 1) #original_param.min()
            max_val = np.percentile(original_param_np, 99) #original_param.max()
            scale = (max_val - min_val) / num_levels
            zero_point = ((-min_val) / scale).round().astype(int)

            # Quantize the parameter
            quantized_param = ((original_param - min_val) / scale + zero_point).round().int()

            # Dequantize the parameter
            dequantized_param = (quantized_param.float() - zero_point) * scale + min_val

            # Update the parameter in the model
            param.data = dequantized_param
    return q_model
