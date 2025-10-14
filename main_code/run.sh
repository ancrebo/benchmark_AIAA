source /scratch/andrescb/condainstall/miniconda3/bin/activate

conda activate /scratch/andrescb/condainstall/miniconda3/envs/condatf_2_10

echo "Running..."
nohup sh -c 'python configurations/configure_problem.py && python read_database.py && python train_database.py' > log_output.log &
