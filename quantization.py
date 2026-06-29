import numpy as np
def compute_scale_zeropoint(tensor, num_bits=8, symmetric=False):
    qmin = -(2 ** (num_bits - 1))      # -128 for 8 bits
    qmax = (2 ** (num_bits - 1)) - 1   # 127 for 8 bits
    
    if symmetric:
        abs_max = np.abs(tensor).max()
        scale = abs_max / qmax
        zero_point = 0
    else:
        # asymmetric
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = int(np.round(qmin - tensor.min() / scale))
    
    return scale, zero_point

def quantize(tensor, scale, zero_point, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    #apply quantization formula
    q = np.round(tensor / scale) + zero_point
    
    # clamp to valid range
    q = np.clip(q, qmin, qmax)
    
    
    return q.astype(np.int8)

def dequantize(q_tensor, scale, zero_point):
    return (scale * (q_tensor.astype(np.float32) - zero_point)).astype(np.float32)

