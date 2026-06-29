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

if __name__ == "__main__":
    # Test 1: asymmetric weight tensor
    weights = np.array([-1.5, -0.8, 0.2, 1.2])
    s, z = compute_scale_zeropoint(weights, symmetric=False)
    print(f"Asymmetric: scale={s:.5f}, zero_point={z}")
    # Expected: scale ≈ 0.01059, zero_point = 14
    
    # Test 2: symmetric weight tensor
    s, z = compute_scale_zeropoint(weights, symmetric=True)
    print(f"Symmetric:  scale={s:.5f}, zero_point={z}")
    # Expected: scale ≈ 0.01181, zero_point = 0
    
    # Test 3: post-ReLU activation (one-sided)
    activations = np.array([0.0, 0.5, 2.3, 4.7, 6.0])
    s, z = compute_scale_zeropoint(activations, symmetric=False)
    print(f"Activation: scale={s:.5f}, zero_point={z}")
    # Expected: scale ≈ 0.02353, zero_point = -128