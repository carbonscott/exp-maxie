#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --partition=milano
#!SBATCH --qos={{ qos }}        # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time {{ walltime }}          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --job-name={{ job }}
#SBATCH --exclusive
#SBATCH --nodes={{ sbatch_num_nodes }}
#SBATCH --ntasks-per-node=1   # One ray instance on a node
#SBATCH --cpus-per-task={{ sbatch_num_cpus }}

cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-maxie/preprocess

export USES_MULTI_NODES=true

echo "NODES: $SLURM_JOB_NODELIST"

# Get the list of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# First node is the head node
head_node=${nodes_array[0]}
echo "Head node: $head_node"

# Extract IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname -I | awk '{print $1}')
echo "Head node IP: $head_node_ip"

# Start Ray on the head node
port=6379
ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus=$SLURM_CPUS_PER_TASK --block --dashboard-host 0.0.0.0 &

# Wait for the head node to initialize
sleep 30

echo "Starting worker nodes..."
# Start Ray on worker nodes
for node in "${nodes_array[@]:1}"; do
    echo "Starting worker node on $node"
    echo "ray start --address=$head_node_ip:$port --num-cpus=$SLURM_CPUS_PER_TASK --block"
    srun --nodes=1 --ntasks=1 -w $node ray start --address=$head_node_ip:$port --num-cpus=$SLURM_CPUS_PER_TASK --block --dashboard-host 0.0.0.0 &
done

sleep 30

# Run the Python script
echo "Running Python script..."
python preprocess_psana_runs.ray.py --multirun exp={{ exp }} 'run=range(1,{{ num_runs }})' num_tasks={{ sbatch_num_tasks }} detector_name={{ detector_name }}

# Shutdown Ray on all nodes
echo "Shutting down nodes..."
for node in "${nodes_array[@]}"; do
    echo "Shutting down Ray on $node"
    srun --nodes=1 --ntasks=1 -w $node ray stop &
done

sleep 30

echo "Job completed"

exit
