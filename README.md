# Edge Detection in 3D Point Clouds using KPConv

This repository contains the implementation of my project for the **ABC Challenge**, focused on **automatic edge detection** in 3D point clouds of manufactured objects using the **Kernel Point Convolution (KPConv)** architecture.

---

## 📘 Project Overview

The goal of the project is to detect **edge points** in 3D point clouds — points that lie on geometric discontinuities such as sharp edges or boundaries of surfaces.  
This task is crucial for applications in **Computer-Aided Design (CAD)**, **reverse engineering**, and **manufacturing inspection**.

The challenge dataset, derived from the **ABC Dataset**, provides for each model:
- `.ply` → Point coordinates (x, y, z) and normals (nx, ny, nz)
- `.lb` → Binary labels (0 = non-edge, 1 = edge)
- `.ssm` → 320 multi-scale geometric descriptors

---

## 🚀 Method

We use **KPConv (Kernel Point Convolution)**, a deep learning method designed for **direct processing of point clouds** without voxelization or projection.

### 🧠 Why KPConv
- Traditional CNNs require regular grids → not suitable for unordered 3D data.  
- KPConv defines **convolution kernels** as points in 3D space around each input point.  
- Each kernel learns to capture **local geometric patterns**.  
- It is efficient, flexible, and achieves excellent results for **segmentation and classification** tasks.

In this project, we use the **rigid KPConv** variant, which offers a good balance between speed, stability, and precision.

---

## 🧹 Data Preprocessing

1. **Integrity checks** between `.ply`, `.lb`, and `.ssm` files  
2. **Z-score normalization** of SSM descriptors  
3. **Grid subsampling** (voxel-based) to reduce density  
4. **Neighbor search** within a given radius (C++ optimized)  
5. **Caching** of all preprocessing results for faster re-runs  

---

## 🏗️ Model Architecture

- U-Net style **encoder–decoder** based on KPConv  
- **Encoder:** KPConv blocks + downsampling  
- **Bottleneck:** global feature aggregation  
- **Decoder:** upsampling + skip connections  
- **Head:** binary classifier for edge vs non-edge

### ⚙️ Key Hyperparameters
| Parameter | Value | Description |
|------------|--------|-------------|
| `num_kernel_points` | 15 | Points per KPConv kernel |
| `first_subsampling_dl` | 0.5 | Initial voxel size |
| `layer_multipliers` | [1, 2, 4] | Multi-scale hierarchy |
| `optimizer` | AdamW | Robust adaptive optimizer |
| `learning_rate` | 1e-3 | Cosine decay schedule |
| `loss` | Weighted CrossEntropy + Dice | Handle class imbalance |

---

## 🧪 Training Pipeline

- Data loading handled by custom `ABCDataset` and `abc_collate`
- Mixed precision (AMP) for faster training
- Gradient clipping for stability
- Weighted loss to handle imbalance (edge points ≈ 6%)
- Metrics: **Precision, Recall, F1, IoU, MCC**

---

## 📊 Results

| Metric | Score |
|---------|-------|
| **Precision** | 0.947 |
| **Recall** | 0.879 |
| **F1-score** | 0.919 |
| **MCC** | 0.915 |
| **Accuracy** | 0.990 |

Training time: ~130s per epoch on a GPU (Google Colab).  
These results demonstrate that KPConv is highly effective for fine geometric edge detection.

---

## 🔮 Future Work

- Data augmentation (rotations, jittering, mirroring)  
- Deformable KPConv variant  
- Automated hyperparameter optimization  
- Model compression (quantization / pruning)


## 📂 Repository Structure

