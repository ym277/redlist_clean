#!/bin/bash
echo "starting at `date` on `hostname`"



# tree_ids=(9 8 7 6 4)
# tree_ids=({13..50})

# tree_ids=($(seq 2001 1 2899))


# mkdir -p slurm_MIP/improved/per_tree/singles/
# mkdir -p res_MIP/improved/per_tree/

job_name="main"
output_log="slurm/${job_name}.out"
error_log="slurm/error/${job_name}.err"

# echo "--output="$output_log" --error="$output_log" --job-name=$tree_id  opt.sh $thres $tree_id $mode $gap"
    # sbatch --output="$output_log" --error="$output_log" --job-name=ym_i"$tree_id"  opt_world.sh $thres $tree_id $mode
# sbatch --output="$output_log" --error="$output_log" --job-name=$tree_id  opt.sh $thres $tree_id $mode $gap
echo "--output="$output_log" --error="$output_log" --job-name=$job_name  opt.sh $job_name"
sbatch --output="$output_log" --error="$output_log" --job-name=$job_name  opt.sh $job_name

