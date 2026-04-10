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

1.  **Image Enhancement:** Uses `CLAHE` for local contrast normalization and `Fast Non-Local Means Denoising` for artifact removal.
2.  **Gradient Analysis:** Employs GPU-accelerated **Scharr Operators** for high-sensitivity edge detection.
3.  **Agent Propagation:** Thousands of independent agents are deployed from seed points, propagating with a dynamic radius based on local gradient magnitude.
4.  **Morphological Refinement:** Agent footprints are processed using `Distance Transform` and `Otsu’s Thresholding` to create the final mask.

---

## 📊 Results and Visualization

The model's performance is analyzed by comparing predictions against expert-annotated **Ground Truth (GT)** masks.

### Detailed Performance Analysis
* 🟡 **Yellow:** Overlap / Intersection (Correct Prediction)
* 🔴 **Red:** False Positive (Over-segmentation)
* 🟢 **Green:** False Negative (Missed fracture area)

<p align="center">
  <img src="/showcase_1.png" alt="Bone Fracture MAS Segmentation Performance Showcase" width="85%" />
</p>
<p align="center">
  <img src="/showcase_2.png" alt="Bone Fracture MAS Segmentation Performance Showcase" width="85%" />
</p>

> **Note:** As seen in the Differential Performance Analysis panel, the system achieves a very high Dice score (0.9854 in this example), with almost no False Positives or False Negatives. Ensure the image is uploaded to the `github_showcase` folder.


## 💻 Installation and Usage

### Requirements
```bash
pip install torch opencv-python numpy pandas pycocotools tqdm matplotlib tabulate
