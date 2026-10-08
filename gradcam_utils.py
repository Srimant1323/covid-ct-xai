"""Helpers for the COVID-19 CT Streamlit app: preprocessing, input check, Grad-CAM, overlay.

The preprocessing mirrors the training pipeline (grayscale -> 224x224 antialiased resize ->
3 identical channels, pixel values kept in 0-255; each model normalises internally).
"""
import numpy as np
import matplotlib
import tensorflow as tf
import keras

IMG_SIZE = 224


def preprocess(pil_img):
    """PIL image -> float32 tensor (224, 224, 3) with values in 0-255."""
    arr = np.array(pil_img.convert("L"), dtype=np.float32)[..., None]
    x = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE), antialias=True)
    x = tf.image.grayscale_to_rgb(x)
    return tf.cast(x, tf.float32)


def check_input(gray):
    """Heuristic check that an image resembles the lung-extracted training images.

    gray: (H, W) array with values 0-255. Returns (ok, list_of_problems).
    This is a rough sanity check, not a guarantee.
    """
    fg = float((gray > 10).mean())                       # fraction of non-black pixels
    edge = np.concatenate([gray[:8].ravel(), gray[-8:].ravel(),
                           gray[:, :8].ravel(), gray[:, -8:].ravel()])
    dark_edge = float((edge <= 10).mean())               # fraction of black border pixels
    problems = []
    if dark_edge < 0.90:
        problems.append("the image border is not black (the model was trained on lung-only "
                        "images with a black background)")
    if fg < 0.02:
        problems.append("almost no lung tissue is visible")
    elif fg > 0.65:
        problems.append("tissue fills most of the image (probably a full, un-segmented CT slice)")
    return len(problems) == 0, problems


def predict_prob(model, x):
    """Probability of the COVID-pattern class for one preprocessed image."""
    return float(model(tf.expand_dims(x, 0), training=False)[0, 0])


def make_gradcam(model):
    """Build a Grad-CAM function for a model made of: preprocessing -> backbone -> pooling -> dense."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    i = model.layers.index(base)
    pre = [l for l in model.layers[:i] if not isinstance(l, keras.layers.InputLayer)]
    head = model.layers[i + 1:]
    dense = head[-1]

    def run(img, sign=1):
        """img: (224, 224, 3) in 0-255. sign=+1: evidence for COVID pattern; -1: for normal.
        Returns (heatmap in 0-1 of shape (224, 224), probability of COVID pattern)."""
        x = tf.expand_dims(tf.cast(img, tf.float32), 0)
        for l in pre:
            x = l(x)
        with tf.GradientTape() as tape:
            feat = base(x, training=False)
            tape.watch(feat)
            z = feat
            for l in head[:-1]:
                z = l(z)
            logit = tf.matmul(z, dense.kernel) + dense.bias      # score before the sigmoid
            target = sign * logit
        grads = tape.gradient(target, feat)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE), method="bilinear")[..., 0]
        return cam.numpy(), float(tf.sigmoid(logit)[0, 0])

    return run


def overlay(gray, cam, alpha=0.45):
    """Blend a 0-1 heatmap (jet colours) over a grayscale image; returns uint8 RGB (H, W, 3)."""
    g = gray / max(float(gray.max()), 1.0)
    base = np.stack([g, g, g], axis=-1)
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
