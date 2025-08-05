# SARL

This project is the algorithm designed to solve the join optimization problem of assortment and caching. 

## Core Features

-   **Dynamic Data Simulation**: A dedicated script generates realistic user request data. Item popularity follows a lifecycle of growth, peak, and decay, with added periodic fluctuations and noise.
-   **PPO-based RL Agent**: The caching agent is trained using the Proximal Policy Optimization (PPO) algorithm, implemented with the framework.
-   **Modular and Configurable**: Key parameters for both data generation and RL training can be easily configured via command-line arguments and configuration classes.

## System Workflow

1.  **Data Generation (`data_generator.py`)**: This script simulates and generates a dataset of user requests over a specified time horizon `T`. It outputs a `records.pkl` file containing the request log and a `popularity_trends.png` visualization.
2.  **RL Training (`train.py`)**: This script uses the `records.pkl` file to drive the RL environment.
    -   It initializes a `DynamicCacheEnv` where the agent's goal is to maximize a reward signal.
    -   The reward is a function of cache hit profit, cache update costs, and a penalty for "thrashing" (frequently adding/removing the same item).
    -   The agent learns a policy to decide which items to place in the cache with the optimal assortment.
    -   Training progress, logs, and the final trained models are saved.

## Installation


 **Create and activate a virtual environment:**
    The required dependencies are listed as follows.
    ```
    gym  >=  0.26.2
    numpy  >=  1.26.4
    python >= 3.12.9
    torch  >= 2.6.0
    tianshou  >=  1.1.0
    tensorboard  >= 2.19.0
    ```


## Usage

Follow these steps to generate data and train the agent.

### Step 1: Generate Simulation Data

Run the data generation script. You can use the default parameters or specify your own.

```bash
python data_generator.py --users 20 --items 100 --time_slots 50 --requests 5000 --output records.pkl
```

-   `--users`: Number of unique users.
-   `--items`: Number of unique items.
-   `--time_slots`: Total number of time slots `T` for the simulation.
-   `--requests`: Total number of user requests to generate.
-   `--output`: The name of the output data file.

This will create `records.pkl` and `popularity_trends.png`.

### Step 2: Train the RL Agent

Once `records.pkl` is generated, you can start training the agent.

```bash
python train.py --gamma your parameter --epoch your parameter --run-name example-v1
```

**Required Arguments:**
-   `--gamma <float>`: The discount factor for future rewards.
-   `--epoch <int>`: The total number of training epochs.

**Optional & Recommended Arguments:**
-   `--run-name <string>`: A unique name for the experiment. All logs and models will be saved under directories with this name. Defaults to a timestamp.
-   `--resume`: Resume training from the last saved checkpoint (`checkpoint.pth`).
-   `--resume-best-weights`: Start a new run but initialize the model with the best-performing weights from a previous run.

## Project Structure

```
.
├── module/
│   └── assortment.py         # Assortment optimization function
├── logs/                     # Directory for TensorBoard logs, run logs, and metrics
├── models/                   # Directory for saved model weights
├── data_generator.py # Script to generate user request data
├── train.py         # Main script to train the PPO agent
└── README.md                 # This file
```

## Configuration & Customization

-   **Data Generation**: To fine-tune the popularity curves, you can modify the "Control Parameters" (`fluctuation_period`, `decay_rate`) directly within the `_generate_item_profiles` and `_calculate_popularity_trends` functions in `data_generator.py`.
-   **RL Training**: For deep configuration of the system (e.g., cache size `C_cs`, number of recommendations `K`, item sizes `s_i`), modify the `SystemConfig` class inside `train.py`.

## Expected Outputs

After a training run, the following will be generated:

-   **Log Directory (`./logs/<run-name>/`)**:
    -   `*.log`: A detailed text log of the training run.
    -   TensorBoard event files for visualizing learning curves (`tensorboard --logdir=./logs`).
-   **Model Directory (`./models/<run-name>/`)**:
    -   `best_policy.pth`: The model weights that achieved the highest reward during evaluation. **This is your final trained model.**
    -   `final_policy.pth`: The model weights at the very end of training.
    -   `checkpoint.pth`: The latest training state for resuming.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.