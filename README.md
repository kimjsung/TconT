# TconT

TconT is a tensor contraction workspace built around backend-specific planning and execution.

Current repository status:
- `tcont` shared library builds successfully with CMake.
- `bench_tccg` benchmark builds and runs through the `plan -> prepare -> benchmark` flow.
- The benchmark supports both FP64 and FP32 benchmark case sets.
- The benchmark can optionally run correctness verification with `--verify`.
- The repository includes a batch runner that executes all TCCG equations and stores per-precision CSV results.

## Requirements

- CMake `>= 3.18`
- A CUDA toolkit installation visible to CMake
- A C++17 compiler
- OpenMP support
- NVIDIA GPU architecture currently configured for `sm_89`

The top-level build currently expects:
- `find_package(CUDAToolkit REQUIRED)`
- `find_package(OpenMP REQUIRED)`

For JSON parsing in the Cogent backend:

- if `nlohmann_json` is available as a CMake package, it is linked normally
- otherwise the build falls back to the vendored headers under `external/nlohmann/`

## Build

Configure and build everything:

```bash
cmake -S . -B build
cmake --build build -j
```

Build only the main library:

```bash
cmake --build build --target tcont -j
```

Build only the TCCG benchmark:

```bash
cmake --build build --target bench_tccg -j
```

If you delete `build/`, recreate it from the repository root:

```bash
cmake -S . -B build
cmake --build build --target bench_tccg -j
```

## Output Layout

Important build outputs:

- Shared library: `build/src/libtcont.so`
- TCCG benchmark: `build/benchmarks/tccg/bench_tccg`
- Batch runner input binary: `build/benchmarks/tccg/bench_tccg`

## Benchmark Layout

Benchmarks are organized by family under `benchmarks/`:

- `benchmarks/tccg/`
- `benchmarks/ccsd/`
- `benchmarks/tal_sh/`
- `benchmarks/tensor_sw/`

The TCCG benchmark is already split by responsibility:

- `tccg_cases.*`: benchmark case definitions
- `tccg_utils.hpp`: printing and size helpers
- `tccg_init.hpp`: benchmark tensor initialization
- `tccg_verify.*`: correctness checking and tolerance handling
- `bench_tccg.cpp`: runner
- `run_tccg_benchmarks.py`: full FP64/FP32 sweep and CSV export

## Run `bench_tccg`

Basic usage:

```bash
./build/benchmarks/tccg/bench_tccg -b 12
```

Show help:

```bash
./build/benchmarks/tccg/bench_tccg --help
```

Run with correctness verification:

```bash
./build/benchmarks/tccg/bench_tccg -b 12 --verify
```

Run the FP32 case set:

```bash
./build/benchmarks/tccg/bench_tccg -b 12 --fp32 --verify
```

Current CLI options:

- `-b`, `--benchmark <id>`: select a TCCG benchmark case
- `--fp64`: run the FP64 benchmark cases
- `--fp32`: run the FP32 benchmark cases
- `--verify`: run correctness verification after timing
- `-h`, `--help`: print usage

Each benchmark run ends with a machine-readable summary line:

```text
TCCG_RESULT equation=12 precision=fp32 operations=... time_ms=... gflops=... validation=PASS
```

## Timing Flow

The benchmark uses the following execution flow:

1. `TconT::plan(desc)`
2. `TconT::prepare(plan)`
3. `TconT::benchmark(run, desc, options)`

The benchmark path performs:

- warm-up launches
- repeated timed launches using CUDA events
- average kernel time reporting
- GFLOPS reporting
- final summary-line reporting for script parsing

Current default benchmark options in `benchmarks/tccg/bench_tccg.cpp`:

- `warmup = 3`
- `repeats = 10`

## Verification

When `--verify` is enabled, the benchmark:

1. runs the selected case
2. copies the output tensor back to host memory
3. dispatches to `verify_tccg_case(...)`

Verification is scalar-type aware:

- `Float64`: strict tolerance
- `Float32`: looser tolerance intended for TF32-style execution

Current verification tolerances:

- `Float64`: `abs_tol=1e-11`, `rel_tol=1e-9`
- `Float32` / TF32-style: `abs_tol=2e-3`, `rel_tol=5e-2`

Verification returns a structured pass/fail result that is emitted by `bench_tccg` as `validation=PASS`, `FAIL`, or `SKIP`.

## Run All TCCG Benchmarks

Use the batch runner to execute every equation for both precisions and save one CSV per precision:

```bash
python3 benchmarks/tccg/run_tccg_benchmarks.py
```

Default outputs:

- `benchmarks/tccg/results/tccg_fp64_results.csv`
- `benchmarks/tccg/results/tccg_fp32_results.csv`

Useful options:

- `--binary <path>`: override the `bench_tccg` binary path
- `--output-dir <dir>`: choose a different results directory
- `--start <id>`: first equation index to run
- `--end <id>`: last equation index to run, inclusive
- `--no-verify`: skip correctness checking

Each CSV contains:

- `equation`
- `precision`
- `operation_count`
- `time_ms`
- `gflops`
- `validation`
- `error_message`

## Current Limitations

- TCCG benchmarking and verification paths support both `ScalarType::Float32` and `ScalarType::Float64`.
- FP32 execution currently follows the Cogent TF32-style generation path and uses correspondingly looser validation tolerances.
- Full benchmark execution still requires a CUDA-capable GPU and a CUDA toolkit visible to CMake.

## Key Source Files

- Core API: `include/tcont.hpp`
- Core implementation: `src/tcont.cpp`
- Cogent backend API: `backends/cogent/include/cogent.hpp`
- Cogent backend implementation: `backends/cogent/src/cogent.cpp`
- TCCG benchmark runner: `benchmarks/tccg/bench_tccg.cpp`
- TCCG benchmark batch runner: `benchmarks/tccg/run_tccg_benchmarks.py`

## Project Tree

High-level repository layout:

```text
TconT/
├── CMakeLists.txt
├── README.md
├── include/
│   └── tcont.hpp
├── src/
│   ├── CMakeLists.txt
│   └── tcont.cpp
├── backends/
│   ├── cogent/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── cogent.hpp
│   │   ├── src/
│   │   │   └── cogent.cpp
│   │   ├── helper/
│   │   ├── plan/
│   │   ├── code/
│   │   └── bin/
│   └── ttgt_cutt/
│       ├── CMakeLists.txt
│       └── include/
│           └── ttgt_cutt.hpp
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── tccg/
│   │   ├── CMakeLists.txt
│   │   ├── bench_tccg.cpp
│   │   ├── tccg_cases.cpp
│   │   ├── tccg_cases.hpp
│   │   ├── tccg_utils.hpp
│   │   ├── tccg_init.hpp
│   │   ├── tccg_verify.cpp
│   │   └── tccg_verify.hpp
│   ├── ccsd/
│   ├── tal_sh/
│   └── tensor_sw/
└── build/
```

Role of the main directories:

- `include/`: public API visible to library users
- `src/`: core orchestration code shared by all backends
- `backends/`: backend-specific planning and execution logic
- `benchmarks/`: benchmark families organized by dataset or workflow
- `build/`: out-of-tree CMake build directory

## Backend Architecture

The execution flow is intentionally split into three layers:

1. `plan`
2. `prepare`
3. `launch` / `benchmark`

In practice:

- `TconT::plan(...)` creates an `ExecutionPlan`
- `TconT::prepare(...)` creates an `ExecutionRun`
- `TconT::launch(...)` launches exactly one run
- `TconT::benchmark(...)` performs warm-up and repeated timed launches

This means a backend can:

- do expensive setup during `plan`
- allocate and bind run-specific buffers during `prepare`
- keep `launch` as small as possible

For Cogent, `plan` already includes:

- planner invocation
- generated kernel discovery
- NVRTC compilation when needed
- `CUmodule` / `CUfunction` loading
- launch metadata preparation

## How To Add A Backend

If you want to add a new backend, the cleanest pattern is:

1. Create a new backend directory under `backends/`
2. Add a backend-local CMake target
3. Implement a backend `PlanImpl`
4. Implement a backend `RunImpl`
5. Hook the backend into `TconT::plan(...)`

Recommended directory layout for a new backend named `my_backend`:

```text
backends/my_backend/
├── CMakeLists.txt
├── include/
│   └── my_backend.hpp
└── src/
    └── my_backend.cpp
```

### 1. Add The Backend Directory

At minimum, add:

- `backends/my_backend/CMakeLists.txt`
- `backends/my_backend/include/my_backend.hpp`
- `backends/my_backend/src/my_backend.cpp`

Then register it from the top-level build:

```cmake
add_subdirectory(backends/my_backend)
```

And link it from `src/CMakeLists.txt` the same way the existing backend targets are linked.

### 2. Implement The Backend Interface

The public orchestration layer lives in `include/tcont.hpp`.

Right now, a backend integrates through:

- `TconT::PlanImpl`
- `TconT::RunImpl`

`PlanImpl` is responsible for:

- backend identity
- converting a high-level equation into an executable plan
- producing a `RunImpl` through `prepare()`

`RunImpl` is responsible for:

- one launch
- exposing benchmark verification buffers when needed

The minimal shape looks like this:

```cpp
struct MyBackendPlanImpl final : TconT::PlanImpl {
  TconT::Backend backend() const override;
  std::shared_ptr<TconT::RunImpl> prepare() const override;
};

struct MyBackendRunImpl final : TconT::RunImpl {
  TconT::Backend backend() const override;
  void launch() override;
  const void* input_left_host_data() const override;
  const void* input_right_host_data() const override;
  void copy_output_to_host(void* destination, size_t bytes) const override;
};
```

### 3. Decide What Belongs In `plan` vs `prepare`

Use this rule of thumb:

- Put expensive, reusable work in `plan`
- Put per-run buffer binding in `prepare`
- Put only the real kernel launch in `launch`

Good candidates for `plan`:

- kernel selection
- descriptor creation
- JIT compilation
- module loading
- launch shape calculation
- immutable metadata

Good candidates for `prepare`:

- input/output allocation
- host initialization
- host-to-device copies
- argument pointer assembly

Good candidates for `launch`:

- one backend kernel launch
- nothing else if possible

### 4. Hook The Backend Into `TconT::plan`

Once your backend plan factory exists, update `src/tcont.cpp` so that:

- backend selection chooses the new backend
- the returned `ExecutionPlan` stores your `PlanImpl`

Today `plan(...)` returns Cogent directly, but the intended extension point is there.

### 5. Support Scalar Types Deliberately

The API already carries:

- `TconT::ScalarType::Float32`
- `TconT::ScalarType::Float64`

When adding a backend, decide early:

- whether you support both types
- whether `Float32` means true FP32 or TF32-like execution
- what validation tolerance is appropriate

If a backend does not support a type yet, fail explicitly with a clear error message rather than silently running the wrong path.

### 6. Add Benchmark Coverage

If the backend should be exercised by a benchmark family:

- add or reuse a benchmark runner under `benchmarks/<family>/`
- use `TconT::plan`, `TconT::prepare`, and `TconT::benchmark`
- enable `--verify` when possible

This keeps backend measurement aligned across benchmark families.

## Suggested Backend Checklist

- Build target added under `backends/<name>/`
- Header and source split cleanly
- `PlanImpl` implemented
- `RunImpl` implemented
- `launch()` contains only launch work
- Scalar type handling defined
- Output copy path implemented for verification
- Benchmark runner or test path added
- README updated if the backend introduces special requirements

## Example Session

```bash
cmake -S . -B build
cmake --build build --target bench_tccg -j
./build/benchmarks/tccg/bench_tccg -b 12 --verify
```
