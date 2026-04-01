import torch
# calculate z = x * y where y = a + b

print("\n"*5)
a = torch.tensor([1,2,3], dtype=torch.float16, requires_grad=True)
b = torch.tensor([4,5,6], dtype=torch.float16, requires_grad=True)
y = a + b
x = torch.tensor([7,8,9], dtype=torch.float16, requires_grad=True)
z = x * y

print(z)
print(z.dtype)

print(a.grad_fn)
print(b.grad_fn)
print(y.grad_fn)
print(x.grad_fn)
print(z.grad_fn)
