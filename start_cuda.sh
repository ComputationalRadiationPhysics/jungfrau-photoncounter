#!/bin/bash

set -x

compile_job_id=$(sbatch compile_cuda.sh | tr -d -c 0-9)
sbatch --dependency=afterok:$compile_job_id run_cuda_base_1.sh
sbatch --dependency=afterok:$compile_job_id run_cuda_base_2.sh
sbatch --dependency=afterok:$compile_job_id run_cuda_base_4.sh
sbatch --dependency=afterok:$compile_job_id run_cuda_detailed_4.sh
sbatch --dependency=afterok:$compile_job_id run_cuda_multiple.sh
