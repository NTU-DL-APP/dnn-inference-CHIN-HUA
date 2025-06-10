import numpy as np

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # 改為支援一維或二維，並保持 axis=-1 的計算穩定性
    x = np.asarray(x)
    if x.ndim == 1:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return np.dot(x, W) + b  # 或 x @ W + b

# === Forward pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            act = cfg.get("activation")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)
    return x

# === Inference API ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
