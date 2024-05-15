# Create a random tensor with shape (7, 7).
import torch

firstTensor = torch.rand(7, 7)
print(firstTensor.shape)
secondTensor = firstTensor.t()
print(torch.mm(secondTensor, firstTensor))

randomTensor = torch.rand(1, 7)
randomTensor = randomTensor.t()

secondTensor = secondTensor.t()
finalResult = torch.mm(secondTensor, randomTensor)
print(finalResult)

randomSeed = 0
torch.manual_seed(randomSeed)

firstTensor = torch.rand(7, 7)
print(firstTensor.shape)
secondTensor = firstTensor.t()
print(torch.mm(secondTensor, firstTensor))

randomTensor = torch.rand(1, 7)
randomTensor = randomTensor.t()

secondTensor = secondTensor.t()
finalResult = torch.mm(secondTensor, randomTensor)
print(finalResult)

# 6.Create two random tensors of shape (2, 3) and send them both to the GPU
# (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors
# (this doesn't have to be the GPU random seed).

# torch.manual_seed(1234)
#
# tensor1 = torch.rand(2, 3)
#
# tensor2 = torch.rand(2, 3)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tensor1 = tensor1.cpu()
# tensor2 = tensor2.cpu()
# print(device)
# print(tensor1)
# print(tensor2)

torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
tensor1 = torch.rand(size=(2, 3)).to(device)
tensor2 = torch.rand(size=(2, 3)).to(device)

print(device)

# Shape: refers to the dimensionality of array or matrix
# Rank: refers to the number of dimensions present in tensor
# creating a tensors

# t1 = torch.tensor([1, 2, 3, 4])
# t2 = torch.tensor([[1, 2, 3, 4],
#                    [5, 6, 7, 8],
#                    [9, 10, 11, 12]])
# print("Tensor t1: \n", t1)
# print("\nTensor t2: \n", t2)
#
# print("\nRank of tensor t1:", len(t1.shape))
# print("Rank of tensor t2:", len(t2.shape))
#
# print("\nRank of tensor t1:", t1.shape)
# print("Rank of tensor t2:", t2.shape)

#A 2*3 matrix has been reshaped and transposed to 3*2.
# We can visualize the change in the arrangement of the elements in the tensor in both cases.

# te = torch.tensor([[1, 2, 3, 4],
#                  [5, 6, 7, 8],
#                  [9, 10, 11, 12]])
# print("Reshaping")
# print(te.reshape(6, 2))
#
# print("\nResizing")
# print(te.resize(2, 6))
#
# print("\nTransposing")
# print(te.transpose(1, 0))

# Mathematical Operations on Tensors in PyTorch
# We can perform various mathematical operations on tensors using Pytorch.
# The code for performing Mathematical operations is the same as in the case with NumPy arrays.
# Below is the code for performing the four basic operations in tensors.
t5 = torch.tensor([1, 2, 3, 4])
t6 = torch.tensor([5, 6, 7, 8])
print("tensor2 + tensor1:")
print(torch.add(t5, t6))

print("\ntensor6 - tensor5:")
print(torch.sub(t6, t5))

print("\ntensor * tensor6:")
print(torch.mul(t5, t6))

print("\ntensor6/tensor5:")
print(torch.div(t6, t5))