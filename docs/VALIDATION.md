# Local validation record

Version 0.2.0 was validated on 2026-10-18 using Windows, CPython 3.12,
NumPy 2.2.4, psutil 7.0.0 and MinGW-w64 GCC 14.2.0.

Completed checks:

1. **24 unittest cases passed with native tests enabled.** Coverage includes
   forward/reverse event equivalence, shared cached actions, empty samples,
   distinct seeds, budget calculations, CELF versus direct greedy, streaming
   versus materialized samples, all method interfaces, custom parameter
   validity, matrix limits, edge-list conversion, all generated example
   configurations, exact visited-set DP on cycles, adaptive sample prefixes,
   fixed-assignment sampler estimates, isolated workers, resume and STOP.
2. **12 native/Python streaming comparisons passed**, using six generated
   fixtures and two stream configurations. Selections, marginal gains,
   coverage and sampler counters agree exactly.
3. **Six native/Python forward-bank and evaluation comparisons passed**.
   Trajectory arrays agree exactly; evaluation statistics agree to 1e-13.
4. **Six native/Python row-major full-bank comparisons passed**, including
   selections, marginal counts and training estimates. Each full-bank prefix
   was replayed and extended by one additional component, matching a separately
   constructed Python reference.
5. **All 15 methods completed end to end on both backend configurations**,
   with two workers and a runtime-generated eight-node graph.
   Python-only selection methods remain Python under the C++ configuration;
   their independent forward evaluations use the native evaluator.
6. **The wheel built and installed into an isolated temporary directory.**
   From outside the source checkout, the installed package compiled its native
   sources, completed all 15 jobs, re-evaluated their saved seeds in parallel,
   and ran graph inspection, exact diagnosis and sampler ablation commands.
   The wheel contains the package modules and C++ sources.
7. **Source-content inspection:** no concrete datasets, plotting imports,
   figure assets, compiled binaries or measured-result archives are included.

The heterogeneity example was corrected during validation: permuting a
sink-specific high consumption probability onto a non-sink could violate the
probability constraints. Its generated graph now has variable out-degrees and
no sinks. Validation continues to reject invalid parameters without silently
clipping them.

These checks validate implementation consistency and executable workflows.
They do not reproduce the manuscript's large-network performance results.
The optional SciPy MAT converter was not separately exercised; edge-list to
CSR NPZ conversion was tested. Linux/macOS execution and the GitHub Actions
workflow were not run locally.
