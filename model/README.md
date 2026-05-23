# Model Data Collection

`model` 폴더의 목적은 predictor 학습용 데이터를 모으는 것입니다. 현재 코드는 `cogent`와 `ttgt`를 같은 문제에 대해 모두 실행하고, 공통 equation/problem feature와 backend별 plan/timing feature를 함께 저장합니다.

## 기본 구조

- `bench_rand_problem.cpp`
  임의 problem file 하나를 읽고, feature 추출과 backend 실행을 수행한 뒤 JSON 1개를 출력합니다.
- `generate_training_data.py`
  여러 `.in` 파일을 순회하면서 `cogent`와 `ttgt` 결과를 하나의 joined row로 저장합니다.
- `build_bench_rand_problem.sh`
  현재 TconT 루트 코드를 빌드한 뒤, `model/bench_rand_problem.cpp`를 별도 바이너리로 컴파일합니다.
- `run_generate_training_data.sbatch`
  Slurm에서 빌드와 데이터 수집을 한 번에 수행합니다.

## 입력 위치 가정

기본적으로 `gen_input`은 각 `TconT` 복제본의 바깥에 있다고 가정합니다.

예시:

```text
workspace/
  gen_input/
    rand_problems/
    rand_problems_new/
    rand_problems_new2/
  TconT/
  TconT_copy_1/
  TconT_copy_2/
```

`generate_training_data.py`는 기본적으로 아래 경로를 자동 탐색합니다.

- `../gen_input/rand_problems`
- `../gen_input/rand_problems_new`
- `../gen_input/rand_problems_new2`

## 추천 수집 방식

학습용 predictor는 결국 `cogent`와 `ttgt` 중 어느 쪽이 빠른지 분류하면 되므로, row 하나에 아래 정보가 같이 들어가도록 저장하는 것이 좋습니다.

- 공통 feature
  equation string, mode 구성, unique/internal/output mode 수, tensor size, internal volume, extent 통계 등
- `cogent` feature
  planner가 만든 kernel/block/stage/tile 정보와 timing
- `ttgt` feature
  transpose/gemm 관련 config와 timing
- label
  실제 실행 시간 기준 winner backend

## 생성 결과

출력 디렉터리마다 다음 파일이 생성됩니다.

- `training_data_*.jsonl`
  문제별 joined raw row
- `training_data_*.csv`
  학습에 바로 쓰기 쉽도록 flatten한 테이블
- `training_data_*.manifest.json`
  shard, precision, backend, row 수 등의 메타데이터

## 로컬 실행 예시

```bash
model/build_bench_rand_problem.sh

python3 model/generate_training_data.py \
  --binary build_model_tools/bench_rand_problem \
  --input-root ../gen_input/rand_problems_new2 \
  --output-dir model/results/rand_problems_new2 \
  --precision fp64 \
  --warmup 100 \
  --repeats 200 \
  --verbose
```

## Slurm 실행 예시

```bash
sbatch --export=ALL,INPUT_ROOT=$(pwd)/../gen_input/rand_problems_new2,NUM_SHARDS=4,SHARD_INDEX=0 \
  model/run_generate_training_data.sbatch
```

여러 `TconT` 복제본에서 동시에 돌릴 때는 `SHARD_INDEX`만 다르게 주면 됩니다.

## 다음 단계에서 추천하는 feature 사용 방향

처음 predictor는 classification으로 가는 것이 맞습니다. 우선은 아래 세 그룹을 기본 feature로 쓰는 것이 좋습니다.

- equation/problem 공통 feature
  구조 자체가 backend 선호도를 강하게 결정합니다.
- backend별 planner/config feature
  실제 kernel/transpose/gemm 형태 차이를 직접 반영합니다.
- 실제 timing 기반 label
  `winner_backend`, `winner_speedup`, `time_ratio_ttgt_over_cogent`

모델 학습 단계에서는 먼저 아래 두 버전을 비교하는 것을 추천합니다.

1. 공통 feature만 사용한 baseline classifier
2. 공통 feature + backend config feature를 함께 사용한 classifier

이렇게 하면 config feature가 실제로 예측력을 얼마나 올리는지 분리해서 볼 수 있습니다.
