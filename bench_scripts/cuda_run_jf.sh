#!/bin/bash

set -x

compile_job_id=$(sbatch compile_cuda.sh | tr -d -c 0-9)
sbatch --dependency=afterok:$compile_job_id cuda_photon.sh
sbatch --dependency=afterok:$compile_job_id cuda_energy.sh
sbatch --dependency=afterok:$compile_job_id cuda_clusters.sh
