# Day 02 - Fused Softmax

**Official topic:** [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

Softmax is normally described as several kernels: row maximum, subtraction,
exponentiation, row sum, and division. Triton keeps one padded row resident in
registers so those stages become one program and global memory is touched only
for the input load and output store.

## Challenge

1. Assign one program to each matrix row.
2. Round the column count up to a power of two for `tl.arange`.
3. Load invalid columns as negative infinity so they cannot affect the maximum.
4. Compute stable softmax with `tl.max`, `tl.exp`, and `tl.sum`.
5. Store only valid columns and test irregular widths such as 1003.

The reduction axis exists inside one Triton program, not across the launch
grid. That makes the row width a resource constraint: very wide rows consume
too many registers or exceed the supported block size. The wrapper therefore
checks the width explicitly rather than silently falling back.

Compare against `torch.softmax`, then benchmark bytes moved rather than FLOPs;
softmax is typically bandwidth-sensitive at these shapes.
