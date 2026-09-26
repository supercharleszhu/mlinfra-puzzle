# Day 10 - Block-Scaled Matrix Multiplication

**Official topic:** [Block-Scaled Matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html).

Block scaling assigns a scale factor to a vector of low-precision values along
K. The mathematical product is a sum of independently scaled K blocks. Native
`tl.dot_scaled` lowers packed FP4/FP8 values and preshuffled scales to
Blackwell Tensor Core or AMD CDNA4 instructions.

## Challenge

1. Partition K into 32-element scale blocks.
2. Load one row scale for A and one column scale for B per K block.
3. Broadcast scales over the corresponding operand tile.
4. Accumulate the scaled tile product in FP32.
5. Validate against an explicit PyTorch sum over scaled K blocks.
6. Explain the packed/preshuffled scale layout required by native hardware.

The executable solution intentionally uses FP16 plus explicit scale
broadcasting, so the semantics run and can be tested on H100. Native
`tl.dot_scaled` requires CUDA compute capability 10/11 or AMD gfx950 in the
current official tutorial. On supported hardware, replace explicit
multiplication with packed FP4/FP8 descriptors and `tl.dot_scaled`; do not
pretend the portable kernel exercises fifth-generation Tensor Cores.
