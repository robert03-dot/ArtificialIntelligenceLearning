import pandas as pd
import torch
import numpy

# 00.PyTorch Fundamentals
scalar = torch.tensor(7)
print(scalar)
print(scalar.item())
scalar = torch.tensor([7, 7])
print(scalar.ndim)
print(scalar.shape)

matrix = torch.tensor([[1, 2],
                       [2, 3]])
print(matrix.ndim)
print(matrix[0])
print(matrix[1])
print(matrix.shape)

Tensor = torch.tensor([[[3, 4, 5],
                        [5, 6, 7],
                        [7, 8, 9]]])
print(Tensor)
print(Tensor.shape)
print(Tensor.ndim)

randomTensor = torch.rand(3, 4)
print(randomTensor)
print(randomTensor.ndim)
print(randomTensor.shape)

random_image_size_tensor = torch.rand(size=(224, 224, 4))
print(random_image_size_tensor.ndim, random_image_size_tensor.shape)

zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros.ndim)
print(zeros.shape)

ones = torch.ones(size=(3, 4))
print(ones)
print(ones.ndim)
print(ones.shape)

print(torch.arange(start=1, end=100, step=3))

ten_zeros = torch.zeros_like(input=torch.arange(start=1, end=100, step=3))

float_32_tensor = torch.tensor([1.0, 2.0, 3.0],
                               dtype=float)
print(f"Datatype of tensor is:{float_32_tensor.dtype}")
print(f"Shape of tensor is:{float_32_tensor.shape}")
print(f"The dimension is:{float_32_tensor.ndim}")
print(f"Device tensor is on:{float_32_tensor.device}")

Tensor = torch.tensor([10, 11])
print(Tensor + 10)
print(Tensor - 10)
print(Tensor * 10)
print(Tensor / 10)

print(torch.mul(Tensor, 10))
print(torch.add(Tensor, 10))
print(torch.subtract(Tensor, 10))
print(torch.divide(Tensor, 10))

print(Tensor, "*", Tensor)
print(f"Equals:{Tensor * Tensor}")

matrix1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

matrix2 = torch.tensor([[1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3]])
# value = 0
# for i in range(len(Tensor)):
#     value += Tensor[i] * Tensor[i]
# print(value)
print(torch.matmul(torch.rand(2, 3), torch.rand(3, 2)))

print(torch.mm(matrix2, matrix1))

print(matrix1.t())
print(matrix1.t().shape)
print(matrix1.t().ndim)

print(f"Original shapes:matrix1 = {matrix1.shape}, matrix2 = {matrix2.shape}")
print(f"New shapes:matrix1 = {matrix1.shape}, matrix = {matrix2.t()}")
print(f"Multiplying:{matrix1.shape}@{matrix2.t().shape} <- inner dimensions must match")
print("Output:\n")
output = torch.matmul(matrix1, matrix2.t())
print(output)
print(f"\nOutput shapes: {output.shape}")

# Find the min
print(torch.min(matrix2))
print(matrix2.min())

print(torch.max(matrix2))
print(matrix2.max())

print(torch.sum(matrix2))
print(matrix2.sum())

# Find the mean - note:the torch.mean() function requires a tensor of float32 datatype to work
print(torch.mean(matrix2.type(torch.float32)))

print(Tensor.argmax())

# Reshaping,stacking,squeezing and unsqueezing

# Reshaping-reshapes an input tensor to a defined shape
# View-Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# Stacking-combine multiple tensors on top of each other(vstack) or side by side(hstack)
# Squeeze-removes all '1' dimensions from a tensor
# Unsqueeze-add a '1' to a target tensor
# Permute-return a view of the input with dimensions permuted(swapped) in a certain way

# Reshaping-add an extra dimension
Tensor = torch.arange(1, 10)
print(Tensor)
print(Tensor.shape)
tensorReshaped = Tensor.reshape(9, 1)
print(tensorReshaped)
print(tensorReshaped.shape)
# Change the view
newTensor = Tensor.view(1, 9)
print(newTensor)
newTensor[:, 0] = 5
print(newTensor)
# Stack tensors on top of each other
tensorStacked = torch.stack([Tensor, Tensor, Tensor, Tensor], dim=1)
print(tensorStacked)
# vstack-0d,hstack-1d
# torch.squeeze()-removes all single dimensions from a target tensor
print(f"Previous tensor:{tensorReshaped}")
print(f"Previous shape:{tensorReshaped.shape}")
# Remove extra dimensions from tensorReshaped
tensorSqueezed = tensorReshaped.squeeze()
print(f"\nNew tensor:{tensorSqueezed}")
print(f"New shape:{tensorSqueezed.shape}")
print(tensorReshaped.squeeze().shape)

# torch.unsqueeze()-add a single dimension to a target tensor at a specific dimension
print(f"Previous target:{tensorReshaped}")
print(f"Previous shape:{tensorReshaped.shape}")
# Add an extra dimension with unsqueeze
tensorUnsqueezed = tensorSqueezed.unsqueeze(dim=1)
print(f"\nNew tensor:{tensorUnsqueezed}")
print(f"\nNew shape:{tensorUnsqueezed.shape}")

# torch.permute-rearranges the dimensions of a target tensor in a specified order
tensorOriginal = torch.rand(size=(224, 224, 3))

# Permuted the original tensor to rearrange the axis(or dim) order
tensorPermuted = tensorOriginal.permute(2, 0, 1)
print(tensorPermuted)

# challenge
Tensor = torch.arange(1, 10).reshape(1, 3, 3)
print(Tensor.shape)
print(Tensor[0][2][2])
print(Tensor[:, :, 2])

# NumPy is a popular scientific Python numerical computing library.
# And because of this,PyTorch has functionality to interact with it
# Data in NumPy,want in PyTorch tensor -> torch.from_numpy(ndarray)
# PyTorch tensor->NumPy->torch.Tensor.numpy()
array = numpy.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array)
print(tensor)
print(array.dtype)
# float64 is numpy's default dtype
# float32 is pytorch's default dtype
# !!! when converting numpy->pytorch,pytorch reflects numpy's default dtype of float64 unless specified otherwise

# Change the value of the array.What will this do to 'tensor'?
array = array + 1
print(array)
print(tensor)

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor.dtype)
print(numpy_tensor.dtype)

# change the tensor,what happens to 'numpy_tensor'?
tensor = tensor + 1
print(tensor)
print(numpy_tensor)

# Reproducibility(trying to take random out of random)
# To reduce the randomness in neural networks.pytorch comes the concept of a **random seed**
# Essentially what the random seed does is "flavour" the randomness
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
# Set the random seed
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

# https://pytorch.org/docs/stable/notes/randomness.html

## Running tensors and PyTorch objects on the GPUs(and making faster computations)

# GPUs = faster computations on numbers, thanks to CUDA+NVIDIA+PyTorch working behind the scenes to make everything hunky dory(good).
# Check for GPU access with PyTorch
# Compute Unified Device Architecture-CUDA is a proprietary parallel computing platform
# and application programming interface (API) that allows software to use certain types of
# graphics processing units (GPUs) for accelerated general-purpose processing,
# an approach called general-purpose computing on GPUs (GPGPU).
print(torch.cuda.is_available())
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Count number of devices
print(torch.cuda.device_count())
# Printing tensors(and models) on the GPU
# The reason we want out tensors/models on GPU is because using a GPU results in faster computations

# Create a tensor(default on the CPU)
tensor = torch.tensor([1, 2, 3], device="cpu")

# Tensor not on GPU
print(tensor, tensor.device)
# Move tensor to GPU(if avalaible)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

##4.Moving tensors back to the GPU
# If tensor is on GPU,can't transform it to NumPy
print(tensor_on_gpu.numpy())
# To fix this,we can first set the tensor to the cpu
