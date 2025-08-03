import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from typing import List, Tuple, Dict, Any

""" 
Control Parameters
1. Fluctuation Period: In the _generate_item_profiles function, the fluctuation_period is determined by random.uniform(2, 25).
 For example, to make fluctuations more frequent, reduce this range (e.g., random.uniform(2, 6)). 
For smoother fluctuations, increase the range (e.g., random.uniform(16, 35)). 
You can modify this to a fixed value or sample from a specified distribution to precisely control the experiment.
2. Decay Rate: 
In the _calculate_popularity_trends function, the decay_rate controls how quickly item popularity fades and can be adjusted as needed.
"""

def generate_simulation_data(
    num_users: int,
    num_items: int,
    T: int,
    total_requests: int,
    visualize: bool = True
) -> List[Tuple[int, int, int]]:
    print("Step 1: Generating item profiles...")
    item_profiles = _generate_item_profiles(num_items, T)

    print("Step 2: Calculating popularity trends for all items over time...")
    pop_trends = _calculate_popularity_trends(item_profiles, T)
    
    time_sum_pop = pop_trends.sum(axis=0)
    non_zero_sum_indices = np.where(time_sum_pop > 0)[0]
    pop_trends[:, non_zero_sum_indices] /= time_sum_pop[non_zero_sum_indices]

    if visualize:
        print("Visualizing popularity trends of sample items...")
        _visualize_trends(pop_trends, item_profiles, T)

    print(f"Step 3: Sampling {total_requests} requests based on popularity trends...")
    requests = []
    item_ids = list(range(num_items))

    valid_time_slots = [t for t in range(T) if pop_trends[:, t].sum() > 0]
    if not valid_time_slots:
        print("Warning: No valid time slots with positive popularity. No requests generated.")
        return []

    for _ in range(total_requests):
        t = random.choice(valid_time_slots)
        uid = random.randrange(num_users)
        probabilities_t = pop_trends[:, t]
        sampled_item = random.choices(item_ids, weights=probabilities_t, k=1)[0]
        requests.append((uid, t, sampled_item))

    print("Step 4: Sorting requests by time...")
    requests.sort(key=lambda x: x[1])

    print("Data generation complete.")
    return requests

def _generate_item_profiles(num_items: int, T: int) -> List[Dict[str, Any]]:

    profiles = []
    for i in range(num_items):
        release_t = random.randint(0, T // 4)
        peak_popularity = random.uniform(0.5, 1.0)
        active_lifecycle_len = T - release_t - random.randint(0, T // 5)
        if active_lifecycle_len < 10: active_lifecycle_len = 10
        
        growth_len = int(active_lifecycle_len * random.uniform(0.1, 0.2))
        peak_len = int(active_lifecycle_len * random.uniform(0.3, 0.5))
        
        fluctuation_period = random.uniform(4, 8)
        fluctuation_amplitude = peak_popularity * random.uniform(0.15, 0.4)
        
        profiles.append({
            "id": i,
            "release_t": release_t,
            "peak_popularity": peak_popularity,
            "growth_end_t": release_t + growth_len,
            "peak_end_t": release_t + growth_len + peak_len,
            "fluctuation_period": fluctuation_period,
            "fluctuation_amplitude": fluctuation_amplitude
        })
    return profiles

def _calculate_popularity_trends(item_profiles: List[Dict[str, Any]], T: int) -> np.ndarray:
    pop_trends = np.zeros((len(item_profiles), T))
    
    for i, profile in enumerate(item_profiles):
        for t in range(T):
            pop = 0.0
            if t < profile["release_t"]:
                pop = 0.0
            elif t <= profile["growth_end_t"]:
                progress = (t - profile["release_t"]) / (profile["growth_end_t"] - profile["release_t"] + 1)
                pop = profile["peak_popularity"] * progress
            elif t <= profile["peak_end_t"]:
                pop = profile["peak_popularity"]
            else:
                decay_rate = 0.05 
                time_past_peak = t - profile["peak_end_t"]
                pop = profile["peak_popularity"] * np.exp(-decay_rate * time_past_peak)

            if pop > 0:
                fluctuation = profile["fluctuation_amplitude"] * np.sin(
                    (2 * np.pi / profile["fluctuation_period"]) * t
                )
                noise = np.random.normal(0, pop * 0.05)
                pop += fluctuation + noise
            
            pop_trends[i, t] = max(0, pop)
            
    return pop_trends

def _visualize_trends(pop_trends: np.ndarray, item_profiles: List[Dict[str, Any]], T: int):
    
    print("\n---Inside _visualize_trends ---")
    print(f"Shape of pop_trends array: {pop_trends.shape}")
    total_sum = np.sum(pop_trends)
    max_val = np.max(pop_trends)
    print(f"Total sum of all popularity values: {total_sum}")
    print(f"Maximum popularity value in the entire dataset: {max_val}")
    
    if total_sum == 0:
        print("Error: Popularity data is all zero. Cannot generate plot.")
        return

    plt.figure(figsize=(12, 7))
    plt.title("Popularity Trends of Sample Items")
    plt.xlabel("Time Slot (t)")
    plt.ylabel("Normalized Popularity") 

    num_samples = min(10, len(item_profiles))
    sample_indices = random.sample(range(len(item_profiles)), num_samples)

    print(f"Plotting {num_samples} sample items with indices: {sample_indices}")

    for i in sample_indices:
        item_curve_data = pop_trends[i, :]

        print(f"  - Plotting item {i}: Max value = {np.max(item_curve_data):.4f}, Min value = {np.min(item_curve_data):.4f}")

        max_pop_of_curve = np.max(item_curve_data)
        if max_pop_of_curve > 0:
            normalized_curve = item_curve_data / max_pop_of_curve
        else:
            normalized_curve = item_curve_data

        if i == sample_indices[0]:
            plt.plot(range(T), normalized_curve, label=f'Item {i} (Highlighted)', color='red', linewidth=1.5)
        else:
            plt.plot(range(T), normalized_curve, alpha=0.6)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_image_file = "popularity_trends.png"
    plt.savefig(output_image_file)
    print(f"Visualization saved to '{output_image_file}'")
    plt.close()

# --- main fun ---
if __name__ == '__main__':
    # --- setup command-line argument parser ---
    parser = argparse.ArgumentParser(description="Generate simulated user-item request data with dynamic popularity trends.")
    parser.add_argument('--users', type=int, default=20, help='Number of users ')
    parser.add_argument('--items', type=int, default=100, help='Number of items')
    parser.add_argument('--time_slots', type=int, default=50, help='Number of time slots')
    parser.add_argument('--requests', type=int, default=5000, help='Total number of requests to generate')
    parser.add_argument('--output', type=str, default="records.pkl", help='Output filename for the dataset (default: records.pkl)')
    
    args = parser.parse_args()

    # --- Use parsed arguments ---
    NUM_USERS = args.users
    NUM_ITEMS = args.items
    TIME_SLOTS = args.time_slots
    TOTAL_REQUESTS = args.requests
    OUTPUT_FILENAME = args.output

    print(f"Starting data generation with parameters:")
    print(f" - Users: {NUM_USERS}")
    print(f" - Items: {NUM_ITEMS}")
    print(f" - Time Slots: {TIME_SLOTS}")
    print(f" - Total Requests: {TOTAL_REQUESTS}\n")

    # generate dataset
    dataset = generate_simulation_data(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        T=TIME_SLOTS,
        total_requests=TOTAL_REQUESTS,
        visualize=True
    )
    
    if dataset:
        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n Dataset successfully saved to '{OUTPUT_FILENAME}'")
        print(f"   The file contains {len(dataset)} records.")
    else:
        print("\nNo data was generated, so no file was saved.")

    if dataset:
        print("\n--- Sample of Generated Dataset (first 20 records) ---")
        print("Format: (user_id, time_slot, item_id)")
        for record in dataset[:20]:
            print(record)
    print(f"\nTotal number of generated requests: {len(dataset)}")