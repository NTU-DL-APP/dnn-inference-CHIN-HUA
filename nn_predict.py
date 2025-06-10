import numpy as np

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.atleast_2d(x)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = e / np.sum(e, axis=-1, keepdims=True)
    return s if x.shape[0] > 1 else s[0]

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

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
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# === Inference API ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)