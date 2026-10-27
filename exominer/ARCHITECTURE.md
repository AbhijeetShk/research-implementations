# Architectural Specification: MultiBranchExoMiner

This document details the multi-modal dynamic attention network for exoplanet candidate vetting, expanding upon traditional multi-branch deep networks.

    +-------------------+        +----------------------+
    | Global/Local Flux |        | Global/Local Centroid|
    |   (Shape: [B, C, L])      |   (Shape: [B, C, L'])   |
    +---------+---------+        +----------+-----------+
              |                             |
              v                             v
      [ 1D Conv Blocks ]            [ 1D Conv Blocks ]
              |                             |
              v                             v
      Flux Feature Map             Centroid Feature Map
      Dim: [B, 1600]                Dim: [B, 384]
              |                             |
              +--------------+--------------+
                             |
                             v
           +------------------------------------+
           | Cross-Modal Attention Bottleneck   |
           |  Query  <-- Flux (1600)            |
           |  Key/Value <-- Centroid Proj(1600) |
           +-----------------+------------------+
                             |
                             v
                [ Scalar Feature Concatenation ]
                             |
                             v
                 [ Calibrated Classifier ]

---

## 1. Asymmetric Multi-Modal Feature Projection

The feature extraction backbones yield asymmetric output dimensions due to differing temporal lengths and kernel strides:
- **Flux Branches:** Output representation dimension $D_{\text{flux}} = 1600$
- **Centroid Branches:** Output representation dimension $D_{\text{centroid}} = 384$

Rather than reducing flux resolution to match centroids, we project the centroid feature vector into the flux vector space using a linear mapping:

$$\mathbf{K}_{\text{proj}} = \mathbf{W}_k \cdot \mathbf{x}_{\text{centroid}}, \quad \mathbf{V}_{\text{proj}} = \mathbf{W}_v \cdot \mathbf{x}_{\text{centroid}}$$

where $\mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{1600 \times 384}$.

---

## 2. Cross-Modal Attention Bottleneck

Flux anomalies (e.g., asymmetric dips) query the spatial centroid displacement maps to evaluate whether light photocenter shifts indicate a background eclipsing binary (BEB):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

- **Query ($\mathbf{Q}$):** Derived from the Flux representation ($\mathbf{x}_{\text{flux}}$).
- **Key ($\mathbf{K}$) & Value ($\mathbf{V}$):** Derived from the projected Centroid representations ($\mathbf{K}_{\text{proj}}, \mathbf{V}_{\text{proj}}$).

---

## 3. Epistemic Uncertainty Estimation

During inference, Dropout layers remain active ($M$ stochastic forward passes):

$$\mu_{\text{pred}} = \frac{1}{M} \sum_{m=1}^M \sigma(f(x; \theta_m)), \quad \sigma^2_{\text{epistemic}} = \frac{1}{M} \sum_{m=1}^M (\sigma(f(x; \theta_m)) - \mu_{\text{pred}})^2$$

High epistemic variance ($\sigma^2 > 0.10$) indicates out-of-distribution noise or instrument corruptions requiring expert human vetting.