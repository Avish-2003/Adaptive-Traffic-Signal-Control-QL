"""
================================================================================
Traffic Signal & Green Light Duration Predictor using Reinforcement Learning
Paper Reference: "Traffic signal control for smart cities using reinforcement learning"
                 (Computer Communications, 2020)
================================================================================
This script explicitly predicts both:
 1. The Optimal Signal Phase (North-South Green vs East-West Green)
 2. The Exact Recommended Green Light Duration (in seconds)
for any given traffic condition (Number of cars / flow on NS and EW directions).
"""
import os
import traci
import numpy as np
import matplotlib.pyplot as plt
TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]
# Discrete Duration Actions (in seconds)
DURATIONS = [10, 20, 30, 45, 60]
# Actions: 
# Actions 0..4 -> NS Green for DURATIONS[0..4]
# Actions 5..9 -> EW Green for DURATIONS[0..4]
NUM_ACTIONS = len(DURATIONS) * 2
# State Discretization Levels for Queue Length (0..3)
def get_queue_level(q):
    if q <= 3:
        return 0  # Light
    elif q <= 8:
        return 1  # Moderate
    elif q <= 15:
        return 2  # Heavy
    else:
        return 3  # Severe
def get_state(ns_q, ew_q):
    ns_lvl = get_queue_level(ns_q)
    ew_lvl = get_queue_level(ew_q)
    return ns_lvl * 4 + ew_lvl  # 16 total states (4x4)
def decode_action(action_idx):
    """
    Safely decodes an action index into (phase_str, duration_sec, phase_code).
    Guaranteed to return a valid 3-tuple.
    """
    action_idx = int(action_idx) % NUM_ACTIONS
    if action_idx < len(DURATIONS):
        phase = "NS Green"
        duration = DURATIONS[action_idx]
        phase_code = 0
    else:
        phase = "EW Green"
        duration = DURATIONS[action_idx - len(DURATIONS)]
        phase_code = 2
    return phase, duration, phase_code
# Webster / Paper Adaptive Optimal Green Time Calculation Formula
def calculate_analytical_duration(ns_cars, ew_cars, min_g=10, max_g=60):
    total_cars = ns_cars + ew_cars
    if total_cars == 0:
        return "NS Green", 15, "EW Green", 15
    
    ns_ratio = ns_cars / total_cars
    ew_ratio = ew_cars / total_cars
    
    cycle_time = min(120, max(30, int(1.5 * total_cars + 15)))
    
    ns_green = max(min_g, min(max_g, int(cycle_time * ns_ratio)))
    ew_green = max(min_g, min(max_g, int(cycle_time * ew_ratio)))
    
    return ns_green, ew_green
# Train Q-Table over Traffic Scenarios
def train_q_table(episodes=500):
    Q = np.zeros((16, NUM_ACTIONS))
    ETA = 0.2
    GAMMA = 0.85
    
    print("Training RL Q-Table for Signal Phase & Duration Prediction...")
    for ep in range(episodes):
        # Simulate diverse queue conditions
        ns_q = np.random.randint(0, 25)
        ew_q = np.random.randint(0, 25)
        state = get_state(ns_q, ew_q)
        
        for _ in range(5):
            # Select action
            if np.random.rand() < 0.2:
                action = np.random.randint(0, NUM_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))
                
            phase, duration, _ = decode_action(action)
            
            # Simulate queue reduction based on duration chosen
            cars_cleared = int(duration * 0.4)  # ~0.4 cars per second green
            
            if phase == "NS Green":
                new_ns = max(0, ns_q - cars_cleared)
                new_ew = ew_q + np.random.randint(1, 4)  # opposing accumulates
            else:
                new_ew = max(0, ew_q - cars_cleared)
                new_ns = ns_q + np.random.randint(1, 4)
                
            # Calculate reward (Paper 4-parameter formulation)
            reward = - (0.5 * (new_ns + new_ew) + 0.3 * np.std([new_ns, new_ew]) - 0.2 * cars_cleared)
            
            next_state = get_state(new_ns, new_ew)
            Q[state, action] += ETA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
            
            ns_q, ew_q = new_ns, new_ew
            state = next_state
            print("Training Complete!\n")
    return Q
# Predict function given specific user inputs
def predict_signal_and_duration(Q, ns_cars, ew_cars):
    state = get_state(ns_cars, ew_cars)
    best_action = int(np.argmax(Q[state]))
    predicted_phase, predicted_duration, _ = decode_action(best_action)
    
    ns_anal, ew_anal = calculate_analytical_duration(ns_cars, ew_cars)
    
    print("=" * 60)
    print(f" TRAFFIC CONDITION INPUT: North-South = {ns_cars} cars | East-West = {ew_cars} cars")
    print("=" * 60)
    print(f" [RL Q-Learning Prediction]")
    print(f"   -> Recommended Signal Phase    : {predicted_phase}")
    print(f"   -> Recommended Green Duration  : {predicted_duration} seconds")
    print(f"\n [Paper/Webster Analytical Calculation]")
    print(f"   -> Calculated NS Green Duration : {ns_anal} seconds")
    print(f"   -> Calculated EW Green Duration : {ew_anal} seconds")
    print("=" * 60 + "\n")
    
    return predicted_phase, predicted_duration, ns_anal, ew_anal

# Visualization helper
def plot_prediction_matrix(Q):
    grid = np.zeros((4, 4))
    duration_grid = np.zeros((4, 4))
    
    for ns in range(4):
        for ew in range(4):
            st = ns * 4 + ew
            best_act = np.argmax(Q[st])
            phase, dur, _ = decode_action(best_act)
            grid[ns, ew] = 1 if phase == "NS Green" else 2
            duration_grid[ns, ew] = dur
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(duration_grid, cmap="YlGnBu")
    
    level_names = ["Light (0-3)", "Mod (4-8)", "Heavy (9-15)", "Severe (16+)"]
    ax.set_xticks(range(4)); ax.set_xticklabels(level_names)
    ax.set_yticks(range(4)); ax.set_yticklabels(level_names)
    ax.set_xlabel("East-West Queue Level")
    ax.set_ylabel("North-South Queue Level")
    ax.set_title("Predicted Green Light Duration (seconds) Across Traffic Conditions")
    
    for i in range(4):
        for j in range(4):
             st = i * 4 + j
             phase, dur, _ = decode_action(np.argmax(Q[st]))
             p_str = "NS" if phase == "NS Green" else "EW"
             ax.text(j, i, f"{p_str}\n{dur}s", ha="center", va="center", color="black", fontweight="bold")
            
    plt.colorbar(im, label="Duration (s)")
    plt.tight_layout()
    plt.savefig("predicted_signal_duration_matrix.png", dpi=150)
    plt.close()
    print("Saved predicted duration matrix to predicted_signal_duration_matrix.png")
if __name__ == "__main__":
    Q = train_q_table(episodes=1000)
    
    # Example Interactive Predictions
    print("=== TRAFFIC SIGNAL & DURATION PREDICTION DEMO ===")
    
    # Scenario 1: Heavy NS traffic, light EW traffic
    predict_signal_and_duration(Q, ns_cars=18, ew_cars=3)
    
    # Scenario 2: Equal heavy traffic
    predict_signal_and_duration(Q, ns_cars=12, ew_cars=14)
    
    # Scenario 3: Light NS traffic, heavy EW traffic
    predict_signal_and_duration(Q, ns_cars=2, ew_cars=15)
    plot_prediction_matrix(Q)
