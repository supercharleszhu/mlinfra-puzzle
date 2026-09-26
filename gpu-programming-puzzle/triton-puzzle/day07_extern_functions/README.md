# Day 07 - External Functions

**Official topic:** [External Functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html).

Triton covers common arithmetic directly, but GPU backends also provide
special functions through CUDA libdevice or the corresponding ROCm libraries.
The `triton.language.extra.libdevice` facade selects the correct symbol for
the input type, so the same kernel can call float or double `asin`.

## Challenge

1. Build the familiar masked vector offsets.
2. Load values constrained to asin's real domain `[-1, 1]`.
3. Call `libdevice.asin` inside the JIT kernel.
4. Store the result and validate both float32 and float64.
5. Explain why an explicit external-library path is usually unnecessary.

The exercise uses Triton's default backend library mapping. Supplying custom
bitcode through `extern_libs` is an advanced deployment option and should only
reference reviewed, trusted libraries. The numerical check uses a tighter
tolerance than the lower-precision matrix lessons because both references call
the backend's correctly rounded special-function implementation.

Afterward, inspect other functions exposed by `libdevice`, but keep domain
restrictions and dtype behavior in the host wrapper.
