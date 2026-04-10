# 🦴 Multi-Agent System (MAS) for Bone Fracture Segmentation

This project introduces an innovative approach to detecting and segmenting bone fractures in X-ray images using **Multi-Agent Systems (MAS)** combined with advanced image processing techniques. Unlike traditional deep learning models, this system utilizes stochastic propagation and morphological refinement methods to identify pathological regions.

## 🚀 Key Features

* **Dataset:** FracAtlas (Validated on 717 images).
* **Performance:** Achieved a **Mean Dice Score of 0.8832**.
* **Technology Stack:** Python, PyTorch (GPU Acceleration), OpenCV, COCO Annotations.
* **Methodology:** Stochastic Pathfinding & Morphological Refinement.

---

## 🛠️ Technical Approach

The system follows a multi-stage pipeline for fracture segmentation:

1.  **Image Enhancement:** Uses `CLAHE` (Contrast Limited Adaptive Histogram Equalization) for local contrast normalization and `Fast Non-Local Means Denoising` for artifact removal.
2.  **Gradient Analysis:** Employs GPU-accelerated **Scharr Operators** for high-sensitivity edge detection to identify potential fracture lines.
3.  **Agent Propagation:** Thousands of independent agents are deployed from seed points, propagating with a dynamic radius based on local gradient magnitude.
4.  **Morphological Refinement:** Agent footprints are processed using `Distance Transform` and `Otsu’s Thresholding` to create the final cohesive mask.

---

## 📊 Results and Visualization

The model's performance is analyzed by comparing predictions against expert-annotated **Ground Truth (GT)** masks.

### Differential Performance Analysis
The visualization panel highlights the model's accuracy using the following color codes:
* 🟡 **Yellow:** Overlap / Intersection (Correct Prediction)
* 🔴 **Red:** False Positive (Model error/over-segmentation)
* 🟢 **Green:** False Negative (Missed fracture area)

> **Note:** Ensure that the `github_showcase` folder is uploaded to your repository for the images to render correctly.

| Input X-Ray | Expert Annotation (GT) | MAS Prediction | Differential Analysis |
| :---: | :---: | :---: | :---: |
| ![Input](github_showcase/showcase_v1.png) | ![GT](github_showcase/showcase_v2.png) | ![Pred](github_showcase/showcase_v3.png) | ![Overlay](github_showcase/showcase_v4.png) |

*(Note: You can update the image paths above to match the specific file names in your folder)*

### Dice Score Distribution
The global performance distribution across 717 samples (Global Mean: 0.8832):

![Performance Distribution](github_showcase/global_performance_distribution.png)

---

## 💻 Installation and Usage

### Requirements
Ensure you have the following libraries installed:
```bash
pip install torch opencv-python numpy pandas pycocotools tqdm matplotlib tabulate
