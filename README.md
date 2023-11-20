# Content-Based Image Retrieval (CBIR) System

## Introduction
This repository contains a Content-Based Image Retrieval (CBIR) system that extracts features from a query image and retrieves similar images from a database. The system implements various image features, including color-based, texture-based, shape-based, and deep methods. It also supports feature fusion and dimension reduction.


## Feature Extraction
### Color-Based Features
- RGB Histogram: [color.py](src/color.py)

### Texture-Based Features
- Gabor Filter: [gabor.py](src/gabor.py)

### Shape-Based Features
- Daisy: [daisy.py](src/daisy.py)
- Edge Histogram: [edge.py](src/edge.py)
- Histogram of Gradient (HOG): [HOG.py](src/HOG.py)

### Deep Methods
- VGG Net: [vggnet.py](src/vggnet.py)
- Residual Net: [resnet.py](src/resnet.py)

### Local Descriptors
- Detected Corners Similarity using SIFT and Harris: [local_description.py](local_description.py)

### Feature Fusion
Some features may lack robustness, and feature fusion is implemented for enhancement.
- Feature Fusion: [fusion.py](src/fusion.py)

### Dimension Reduction
To address the curse of dimensionality, random projection is used.
- Random Projection: [random_projection.py](src/random_projection.py)

## Evaluation
The CBIR system evaluates images based on feature similarity using Mean Average Precision (MAP) metrics. The evaluation is implemented in [evaluate.py](src/evaluate.py).

Method | Color | Daisy | Edge | Gabor | HOG | VGG Net | ResNet
--- | --- | --- | --- |--- |--- |--- |---
Mean MAP (depth=10) | 0.614 | 0.468 | 0.301 | 0.346 | 0.450 | 0.914 | 0.944

## How to Run the Code

### Part 1: Building the Image Database
1. Clone the repository.
2. Create a directory named `database` and organize your images into subdirectories based on classes.


### Part 2: Running the Code
- Streamlit User Interface
```
streamlit run gui.py
```
- Alternatively you can run search queries via `query.py` script.

### Data Augmentation
Run the data augmentation script to get more images in the same directory.
```bash
python data_augmentation/augmentor.py <database_path> [flipv, fliph, noise, rot, trans, zoom, blur]
```

### References
- [Original Image Augmentation](https://github.com/codebox/image_augmentor)
- [Original CBIR Project](https://github.com/pochih/CBIR)
