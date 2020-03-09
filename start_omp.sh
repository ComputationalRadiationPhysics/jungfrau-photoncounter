#!/bin/bash

set -x

compile_job_id=$(sbatch compile_omp.sh | tr -d -c 0-9)
sbatch --dependency=afterok:$compile_job_id run_omp_base.sh
sbatch --dependency=afterok:$compile_job_id run_omp_detailed.sh
sbatch --dependency=afterok:$compile_job_id run_omp_multiple.sh
