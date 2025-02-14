# PyTorch Tensor Operations in Google Colab

This repository contains a **Google Colab Notebook** demonstrating various tensor operations using PyTorch. It covers the creation of tensors, manipulation, indexing, slicing, and basic mathematical operations. The notebook also includes examples of converting tensors to NumPy arrays and saving/loading tensors.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [How to Use](#how-to-use)
- [Code Examples](#code-examples)

---

## Introduction

PyTorch is a popular open-source machine learning library that provides a flexible framework for tensor computations and deep learning. This notebook focuses on fundamental tensor operations, which are essential for understanding and working with PyTorch. The notebook is designed to run in **Google Colab**, making it easy to experiment with PyTorch without needing to set up a local environment.

---

## Features

- **Tensor Creation**: Create scalars, vectors, matrices, and N-dimensional tensors.
- **Tensor Manipulation**: Reshape, transpose, and perform element-wise operations.
- **Indexing and Slicing**: Access and modify tensor elements using indexing and slicing.
- **Mathematical Operations**: Perform basic arithmetic, matrix multiplication, and statistical operations.
- **Conversion**: Convert tensors to NumPy arrays and vice versa.
- **Saving and Loading**: Save tensors to disk and load them back.
- **Visualization**: Use Matplotlib to visualize data distributions.

---

## How to Use

### Running the Notebook in Google Colab
1. **Open the Notebook**:
   - Click the **Open in Colab** button below:  
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/your-notebook-name.ipynb)
   - Alternatively, you can upload the notebook to your Google Drive and open it in Colab.

2. **Run the Notebook**:
   - Once the notebook is open in Colab, click **Runtime** in the top menu and select **Run all** to execute all cells.
   - You can also run each cell individually by clicking the play button next to the cell.

3. **Install Dependencies**:
   - If required, the notebook will install PyTorch and other dependencies automatically. If not, you can install them manually by running:
     ```python
     !pip install torch numpy matplotlib
     ```

---

## Code Examples

### Tensor Creation
```python
# Create a scalar
scalar = torch.tensor(4)

# Create a vector
vector = torch.tensor([1, 2, 3, 4])

# Create a matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

### Tensor Manipulation
```python
# Reshape a tensor
reshaped_tensor = tensor.view(2, 3, 2)

# Transpose a matrix
transposed_matrix = matrix.t()
```

### Mathematical Operations
```python
# Element-wise addition
result = tensor1 + tensor2

# Matrix multiplication
result = torch.matmul(matrix1, matrix2)
```

### Indexing and Slicing
```python
# Access elements
element = tensor[0, 1, 2]

# Slice a tensor
sliced_tensor = tensor[1:3, :, 0:2]
```

### Saving and Loading Tensors
```python
# Save a tensor
torch.save(tensor, 'tensor.pt')

# Load a tensor
loaded_tensor = torch.load('tensor.pt')
```

### Visualization
```python
# Plot a histogram of random values
rand_gauss = torch.randn(10000)
plt.hist(rand_gauss, bins=100)
plt.show()
```

---

### Additional Notes
- If you encounter any issues while running the notebook, ensure that you are using a compatible version of PyTorch and other dependencies.
- Feel free to modify the notebook to experiment with different tensor operations and explore PyTorch further.

---
