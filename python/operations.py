from torch import tensor, rand as tensor_rand
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

