from torch import tensor, rand as tensor_rand
import torch 

# element-wise operations
print("\n"*5)

x = tensor_rand((2,3))
y = tensor_rand((3,1))
print(f"x: {x}, \ty: {y}")
# z = x + y
# z = x - y
# z = x * y
# z = x / y
# elemental_mult = x * y
# print(elemental_mult)
matrix_mult = x @ y
print(matrix_mult)

scores = tensor([[10,20,30], [5,10,15]], dtype=torch.float32)
print(f"scores: {scores}")
avg_per_student = scores.mean(dim=1)
avg_per_assignment = scores.mean(dim=0)
print(f"avg_per_student: {avg_per_student}, \navg_per_assignment: {avg_per_assignment}")
# dimension 0 collapses vertically (does operation per column, like mean per column), 
# dimension 1 collapses horizontally(does operation per row, like mean per row), 


scores = tensor([
    [10, 0 , 5 , 20, 1], 
    [1, 30 , 2, 5 ,0]
], dtype=torch.float32)
best_indices = scores.argmax(dim=1)
print(f"best_indices: {best_indices}")
# gives you the position of the max value in each row, so for the first row it gives 3 because 20 is the max value and it's in position 3, for the second row it gives 1 because 30 is the max value and it's in position 1. result: tensor([3, 1])