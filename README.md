# CUDA + GPU Profiling Mini-Project

## Overview
A mini-project exploring CUDA GPU programming and GPU performance profiling using a Jupyter Notebook.

## Objectives
- Implement a tiled matrix multiplication kernel using CUDA C/C++.
- Compile and execute the CUDA kernel using NVIDIA's CUDA compiler (`nvcc`).
- Compare CPU and GPU execution performance and calculate speedup.
- Profile a ResNet-18 model using PyTorch Profiler to identify GPU kernels with the highest execution time.

## Technologies Used
- CUDA C/C++
- NVIDIA GPU and CUDA Toolkit
- Python
- PyTorch and Torchvision
- PyTorch Profiler
- Google Colab / Jupyter Notebook

## How to Run
1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another compatible GPU).
3. Run the notebook cells in order.
4. Review the matrix multiplication correctness check, CPU/GPU speedup, and GPU profiling results.

## Results
Record the measured CPU/GPU speedup, maximum absolute error, and the top GPU kernel from the profiler after running the notebook.

## Project File
- `CUDA_GPU_miniproject.ipynb` — implementation and profiling workflow.

*Note: Actual performance results depend on the GPU, runtime environment, and input configuration.*
