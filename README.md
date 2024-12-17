# Photometric Stereo with RANSAC

This code is for the assignment 3 in the Vision and Image Processing - frasalute is one of the main contributors. All of the assignment questions are answered within. The code is meant to run in one go.

This repository implements photometric stereo techniques for surface normal estimation and depth reconstruction across multiple datasets. The project employs Lambertian reflectance models, robust estimation using RANSAC, and 3D surface visualization.

## Features
- **Surface Normal Estimation**: Calculation of surface normals using photometric stereo techniques.
- **Albedo Map Visualization**: Visualization of albedo (reflectance) across surfaces.
- **Depth Reconstruction**: Integration of surface normals to reconstruct depth maps.
- **Robust RANSAC Estimation**: Improved normal estimation using RANSAC to handle outliers.
- **3D Surface Visualization**: 3D surface plots from multiple viewpoints.

## Datasets
The following datasets are used for evaluation:
- **Beethoven**
- **mat_vase**
- **shiny_vase**
- **Buddha**
- **Face**

## Requirements
The following Python libraries are required:
- `numpy`
- `matplotlib`
- `os`
- `ps_utils` (Custom utility functions for photometric stereo)

Install dependencies using:
```bash
pip install numpy matplotlib
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/frasalute/photometric-stereo.git
   cd photometric-stereo
   ```
2. Ensure the datasets (e.g., `Beethoven.mat`, `mat_vase.mat`) are in the project directory.
3. Run the script:
   ```bash
   python Full_script.py
   ```

## Project Structure
```
.
|-- Full_script.py       # Main script to run all tasks
|-- ps_utils.py          # Utility functions for photometric stereo
|-- datasets/            # Contains .mat dataset files
|-- README.md            # Project documentation
```

## Output
The script outputs:
- Albedo images
- Surface normal components (n1, n2, n3)
- Depth maps with 3D surface visualizations

## Sample Visualization
| **Albedo Map** | **Surface Normals** | **3D Reconstruction** |
|:-------------:|:-------------------:|:---------------------:|
| ![albedo](https://github.com/user-attachments/assets/aea95718-e06a-4c16-93b6-9aeed8510333)| ![normal compo](https://github.com/user-attachments/assets/d4b037e1-fc45-489b-8f36-33a14b2cc032)
 | ![3d](https://github.com/user-attachments/assets/47740e5f-1034-4360-865f-929dc7734b2f) |



