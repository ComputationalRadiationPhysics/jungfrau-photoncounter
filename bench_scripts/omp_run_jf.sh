#!/bin/bash

set -x

compile_job_id=$(sbatch compile_omp.sh | tr -d -c 0-9)
sbatch --dependency=afterok:$compile_job_id omp_photon.sh
sbatch --dependency=afterok:$compile_job_id omp_energy.sh
sbatch --dependency=afterok:$compile_job_id omp_clusters.sh
