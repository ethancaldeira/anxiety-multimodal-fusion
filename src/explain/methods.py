"""
explain.methods — attribution methods with a uniform API.

Output conventions:
  - Grad-CAM and Grad-CAM++ return a non-negative class-activation map at
    the resolution of the target conv layer's feature map, min-max
    normalised to [0, 1] per sample.
  - SmoothGrad returns a gradient-magnitude map with the shape of the
    input, min-max normalised to [0, 1] per sample.
  - Integrated Gradients returns the raw signed attribution with the shape
    of the input (positive values push towards the positive class); it is
    not normalised.

Available methods:
  gradcam(model, x, layer_name=None)         — Grad-CAM (Selvaraju 2017)
  gradcam_plus_plus(model, x, layer_name=None) — Grad-CAM++ (Chattopadhay 2018)
  smoothgrad(model, x, n_samples=25, noise=0.1) — SmoothGrad (Smilkov 2017)
  integrated_gradients(model, x, n_steps=50)    — Integrated Gradients
                                                  (Sundararajan 2017)

Common arguments:
  model       a compiled tf.keras model with a single-logit sigmoid output
  x           input tensor with batch dim, shape (B, ...) or (1, ...)
              For sMRI: (1, H, W, 3) — RGB-replicated MRI slice batch
              For fMRI roi_roi: (1, n_features) — flattened upper-triangle
              For fMRI voxel: (1, X, Y, Z, n_rois) — 3D-CNN input
  layer_name  conv layer to attribute over (Grad-CAM only); if None, the
              convolutional backbone sub-model or the last top-level conv
              layer is used (see _auto_conv_target)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

# SmoothGrad uses tf-keras-vis (see requirements.txt). Grad-CAM, Grad-CAM++
# and Integrated Gradients are implemented below.
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_positive_class(output):
    """Score function for binary sigmoid head: encourage positive logit.

    tf-keras-vis maximises the returned scalar(s) over the batch. With a
    single-logit sigmoid output the activation IS the positive-class
    probability, so we return it directly.
    """
    # output shape (B, 1) — squeeze to (B,)
    return tf.squeeze(output, axis=-1) if output.shape[-1] == 1 else output[:, 1]


def _auto_conv_target(model: tf.keras.Model):
    """Pick the layer in ``model.layers`` whose output gives the
    Grad-CAM feature map.

    Walks the top-level layers in reverse. If a layer is itself a
    sub-model (Functional/Sequential, e.g. an embedded ResNet50), and
    that sub-model contains at least one Conv2D/Conv3D, return the
    sub-model — its output (typically the final feature map of the
    convnet) IS what we want to attribute over, and unlike the
    individual convs inside it, the sub-model's output tensor IS in the
    top-level graph and can be hooked.

    Otherwise return the last Conv2D/Conv3D layer with a feature-map
    output (rank-4 for 2D, rank-5 for 3D).

    Raises ValueError if no conv layer can be found anywhere.
    """
    for layer in reversed(model.layers):
        # Sub-model case (transfer-learning wrapper): if it contains any
        # conv layers, its OUTPUT is the feature map we want.
        if isinstance(layer, tf.keras.Model):
            inner_convs = [s for s in layer.layers
                           if isinstance(s, (tf.keras.layers.Conv2D,
                                             tf.keras.layers.Conv3D))]
            if inner_convs:
                return layer
            continue
        # Flat-model case: a top-level conv with a feature-map output.
        if not isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.Conv3D)):
            continue
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) in (4, 5):
            return layer
    raise ValueError(
        f"Could not auto-find a 2D/3D conv layer in {model.name}; "
        f"pass layer_name explicitly."
    )


def _resolve_target(model: tf.keras.Model, layer_name: str | None):
    """Look up the conv-layer target — either by explicit name (top-level
    only) or via auto-detection."""
    if layer_name is not None:
        return model.get_layer(layer_name)
    return _auto_conv_target(model)


def _build_prober(model: tf.keras.Model, target):
    """Re-trace ``model``'s top-level call chain into a fresh Functional
    that returns ``[feature_map, final_output]``.

    The top-level layers are re-applied to a new ``Input`` so that the
    target's output and the final output belong to one graph, which also
    covers targets that are nested sub-models (e.g. a ResNet50 backbone).

    Assumes ``model.layers`` is in topological order (true for the
    Functional/Sequential models trained by the upstream repositories).
    """
    inp = tf.keras.Input(shape=model.input.shape[1:], name='prober_in')
    x = inp
    fmap = None
    for L in model.layers:
        if isinstance(L, tf.keras.layers.InputLayer):
            continue
        x = L(x)
        if L is target:
            fmap = x
    if fmap is None:
        raise ValueError(f"Target layer {target.name} not found in top-level "
                         f"layer chain of {model.name}.")
    return tf.keras.Model(inp, [fmap, x], name='prober')


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalise to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    mn = flat.min(axis=1, keepdims=True)
    mx = flat.max(axis=1, keepdims=True)
    rng = np.maximum(mx - mn, 1e-12)
    return ((flat - mn) / rng).reshape(arr.shape)


# ---------------------------------------------------------------------------
# Grad-CAM family
# ---------------------------------------------------------------------------

def _gradcam_core(model: tf.keras.Model, x: np.ndarray,
                  layer_name: str | None, plus_plus: bool) -> np.ndarray:
    """Grad-CAM / Grad-CAM++ for flat CNNs and for nested sub-model
    architectures (e.g. a ResNet50 backbone inside a Functional head).

    Implementation:
      * Pick a target layer in ``model.layers`` whose output IS in the
        top-level graph (the sub-model itself for transfer-learning
        wrappers, or the last top-level conv for flat CNNs).
      * Re-trace the top-level call chain through a fresh Functional
        that exposes ``[feature_map, final_output]`` (``_build_prober``).
      * Take gradients of the positive-class score wrt the feature map.
      * Weights = GAP of grads over spatial dims; CAM = weighted sum
        over channels, then ReLU.
      * Grad-CAM++ uses (grads^2) / (2*grads^2 + sum(fmap)*grads^3) for
        per-pixel weights instead of GAP (Chattopadhay 2018 eq. 19).
    """
    target = _resolve_target(model, layer_name)
    prober = _build_prober(model, target)

    x_t = tf.constant(np.asarray(x, dtype=np.float32))
    with tf.GradientTape() as tape:
        fmap, score = prober(x_t, training=False)
        s = (tf.squeeze(score, axis=-1)
             if score.shape[-1] == 1 else score[:, 1])
    grads = tape.gradient(s, fmap)            # same shape as fmap
    spatial_axes = tuple(range(1, len(fmap.shape) - 1))   # all dims between batch and channel

    if not plus_plus:
        # Vanilla Grad-CAM: weights = GAP of gradients.
        weights = tf.reduce_mean(grads, axis=spatial_axes, keepdims=True)
    else:
        # Grad-CAM++ per-pixel weights (Chattopadhay 2018).
        g2 = grads ** 2
        g3 = g2 * grads
        sum_fmap = tf.reduce_sum(fmap, axis=spatial_axes, keepdims=True)
        denom = 2.0 * g2 + sum_fmap * g3
        denom = tf.where(tf.abs(denom) < 1e-12,
                         tf.ones_like(denom), denom)
        alpha = g2 / denom
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads),
                                axis=spatial_axes, keepdims=True)

    cam = tf.reduce_sum(fmap * weights, axis=-1)   # collapse channel
    cam = tf.nn.relu(cam).numpy()
    return _normalize(cam)


def gradcam(model: tf.keras.Model, x: np.ndarray,
            layer_name: str | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al. 2017, ICCV).

    Returns activation map at the resolution of the target conv layer,
    normalised to [0, 1]. The caller is responsible for upsampling to
    input resolution before overlaying.
    """
    return _gradcam_core(model, x, layer_name, plus_plus=False)


def gradcam_plus_plus(model: tf.keras.Model, x: np.ndarray,
                      layer_name: str | None = None) -> np.ndarray:
    """Grad-CAM++ (Chattopadhay et al. 2018, WACV) — sharper localisation
    than vanilla Grad-CAM, especially for multi-object inputs.
    """
    return _gradcam_core(model, x, layer_name, plus_plus=True)


# ---------------------------------------------------------------------------
# Gradient-based saliency
# ---------------------------------------------------------------------------

def smoothgrad(model: tf.keras.Model, x: np.ndarray,
               n_samples: int = 25, noise: float = 0.1) -> np.ndarray:
    """SmoothGrad (Smilkov et al. 2017) — averages |gradient| over
    `n_samples` noisy copies of `x` (Gaussian, std = `noise` × range(x)).
    Reduces high-frequency artefacts in vanilla gradient maps.
    """
    sal = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    attr = sal(_score_positive_class, x,
               smooth_samples=n_samples, smooth_noise=noise)
    return _normalize(np.asarray(attr))


def integrated_gradients(model: tf.keras.Model, x: np.ndarray,
                          baseline: np.ndarray | None = None,
                          n_steps: int = 50,
                          subjects_per_chunk: int = 8) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al. 2017, ICML).

    IG(x) = (x - baseline) × mean over α ∈ [0, 1] of ∂f(baseline + α·(x-baseline))/∂x

    Default baseline is the all-zeros tensor with the same shape as x;
    the path integral uses `n_steps` evenly spaced points from 0 to 1.

    Memory: processes the test batch in chunks of `subjects_per_chunk`
    so the interpolation tensor stays bounded at
    `n_steps * subjects_per_chunk * prod(feature_shape)` floats.
    For an MLP with 19,900 input features on a T4 (≤1 GB free), 8
    subjects/chunk is comfortable.

    Returns the SIGNED attribution (positive = contributes toward
    positive class). The reporting layer takes the absolute value before
    visualising.
    """
    x = np.asarray(x, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(x)
    else:
        baseline = np.asarray(baseline, dtype=np.float32)
        assert baseline.shape == x.shape

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    B = x.shape[0]
    out = np.empty_like(x)

    for start in range(0, B, subjects_per_chunk):
        stop  = min(start + subjects_per_chunk, B)
        x_c   = x[start:stop]
        b_c   = baseline[start:stop]
        # interp shape = (n_steps, chunk, *feat) → flatten to (n_steps*chunk, *feat)
        deltas = (x_c - b_c)[None, ...] * alphas[(slice(None),) +
                                                  (None,) * x_c.ndim]
        interp = (b_c[None, ...] + deltas).reshape((n_steps * x_c.shape[0],)
                                                    + x_c.shape[1:])
        interp_tf = tf.convert_to_tensor(interp)

        with tf.GradientTape() as tape:
            tape.watch(interp_tf)
            preds  = model(interp_tf, training=False)
            scores = _score_positive_class(preds)
        grads = tape.gradient(scores, interp_tf).numpy()
        grads = grads.reshape((n_steps, x_c.shape[0]) + x_c.shape[1:])
        avg_grad = grads.mean(axis=0)
        out[start:stop] = (x_c - b_c) * avg_grad

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dispatch — used by src/report_saliency.py
# ---------------------------------------------------------------------------

METHODS = {
    'gradcam':           gradcam,
    'gradcam_plus_plus': gradcam_plus_plus,
    'smoothgrad':        smoothgrad,
    'integrated_gradients': integrated_gradients,
}


def attribute(method: str, model: tf.keras.Model, x: np.ndarray, **kwargs):
    """Convenience dispatch — `attribute('gradcam', model, x)` etc."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(METHODS.keys())}"
        )
    return METHODS[method](model, x, **kwargs)
