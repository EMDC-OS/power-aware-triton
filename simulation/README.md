# Simulation

## Run
To run the simulation, use the following command:
```bash
python3 Efficient_GPU_alloc_sim.py < log_filename
```

## Parameters
To execute the simulation, a config.txt file must exist in the same directory. The config.txt file contains the parameters necessary for the simulation.

### Config.txt
The following parameters should be defined in `config.txt`:
- List of GPU names
- Allocation priority for each GPU
- Number of GPU servers to use in the simulation
- Maximum power consumption for each model-GPU combination
- Maximum inference throughput for each model-GPU combination
- Number of inference services running in the cloud
- Names of models used by each service
- Filename of the simulation trace

### model_config
`model_config` contains profiling results for each GPU server. If new GPU servers or inference models are added, additional profiling should be conducted to utilize them. There are three available options:

- `default`: Baseline configuration
- `dvfs`: For Approach 4
- `scale`: For Approach 3