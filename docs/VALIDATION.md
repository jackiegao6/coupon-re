# Local validation record

Validated on 2026-10-18 using Windows, CPython 3.12.8, NumPy 2.2.4,
psutil 7.0.0, setuptools 78.1.0, and MinGW-w64 GCC 14.2.0.

The final core sources were copied to a separate directory for verification.
No manuscript files, benchmark datasets, result archives, or plotting scripts
were available in that copy.

Completed checks:

1. **14 unittest cases passed**, with native tests enabled. These include exhaustive
   forward/reverse equivalence; cached actions across branches; retained empty
   samples; unique seeds; the sufficient-budget calculation; full-bank/streaming
   equivalence; CELF/plain-greedy equivalence; graph input validation; behavior
   parameters; parallel orchestration; resume; and protocol mismatch rejection.
2. **12 native/Python streaming comparisons passed**: six generated fixtures and
   two root/component seed configurations. Choices, marginal gains, coverage
   counts, and RR counters agree exactly.
3. **Six forward-bank/evaluation comparisons passed**. Python and C++ trajectory
   arrays agree exactly; independently evaluated statistics agree to 1e-13.
4. **The complete seven-method demo passed on both backends** with two job workers.
   All selected seeds match; spread, redeemed coupons, evaluation variance, and
   confidence halfwidth agree to 1e-12.
5. **The wheel built and installed successfully**. Native sources are present in
   the installed package. Running the C++ demo from outside the source checkout
   also completed all seven jobs.
6. **Upload-content inspection passed**: no datasets, plots, native executables,
   machine-specific absolute paths, or old experiment result files are included.

The independent tests demonstrate implementation consistency and a runnable
workflow. They do not constitute large-network performance results or numerical
reproduction of the manuscript's figures. Linux/macOS execution and the GitHub
Actions workflow have not been run locally.
