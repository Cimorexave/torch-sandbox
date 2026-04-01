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
