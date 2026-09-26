# Day 11 - Programmatic Dependent Launch

**Official topic:** [Programmatic Dependent Launch](https://triton-lang.org/main/getting-started/tutorials/11-programmatic-dependent-launch.html).

CUDA programmatic dependent launch lets a dependent grid begin before every
CTA in the prior grid has retired. Triton exposes grid dependency control
through `tl.extra.cuda.gdc_wait()` and
`tl.extra.cuda.gdc_launch_dependents()`, while the launch itself must opt in
with `launch_pdl=True`.

## Challenge

1. Add a compile-time `USE_GDC` path around the two device operations.
2. Wait before consuming memory produced by an earlier dependent grid.
3. Announce dependent work after this program no longer needs launch-blocking
   resources.
4. Pass the same boolean to the kernel meta-parameter and launch option.
5. Compare enabled and disabled outputs on compute capability 9 or newer.

`gdc_launch_dependents` is a scheduling hint, not a memory fence.
`gdc_wait` supplies the ordering point before dependent memory is consumed.
The numerical operation remains vector addition so correctness cannot obscure
the launch semantics. Benchmarking should use a chain of kernels; an isolated
single launch has no dependency overlap to exploit.
