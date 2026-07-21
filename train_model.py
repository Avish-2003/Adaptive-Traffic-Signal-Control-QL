"""
Multi-episode Q-learning training for the 2-direction intersection.
Builds on your existing reward function, extended with:
  - multiple training episodes (not just one pass)
  - epsilon decay (explore a lot early, exploit more later)
  - a frozen evaluation phase after training
  - per-episode reward logging for a convergence plot
"""

import traci
import numpy as np
import csv
import os

TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]

STATE_NS_GREEN  = "GGggrrrrGGggrrrr"
STATE_NS_YELLOW = "yyyyrrrryyyyrrrr"
STATE_EW_GREEN  = "rrrrGGggrrrrGGgg"
STATE_EW_YELLOW = "rrrryyyyrrrryyyy"

ETA, GAMMA = 0.1, 0.9
EPISODES = 60
START_EPSILON = 1.0
END_EPSILON = 0.05
EPSILON_DECAY = 0.94

GREEN_TIME, YELLOW_TIME, SIGNAL_LEN = 10, 3, 10
SUMOCFG = "sim.sumocfg"

W_THROUGHPUT, W_QUEUE, W_WAIT, W_FAIRNESS = 0.55, 0.20, 0.15, 0.10
MAX_QUEUE, MAX_THROUGHPUT, MAX_WAIT, MAX_FAIRNESS = 9, 8, 10.3, 4.0


def build_custom_program():
    phases = [
        traci.trafficlight.Phase(GREEN_TIME, STATE_NS_GREEN),
        traci.trafficlight.Phase(YELLOW_TIME, STATE_NS_YELLOW),
        traci.trafficlight.Phase(GREEN_TIME, STATE_EW_GREEN),
        traci.trafficlight.Phase(YELLOW_TIME, STATE_EW_YELLOW),
    ]
    logic = traci.trafficlight.Logic("custom", 0, 0, phases)
    traci.trafficlight.setProgramLogic(TLS_ID, logic)
    traci.trafficlight.setProgram(TLS_ID, "custom")


def get_queues():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return ns, ew


def get_avg_waiting_time():
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        return 0.0
    return sum(traci.vehicle.getWaitingTime(v) for v in veh_ids) / len(veh_ids)


def get_state(ns, ew):
    return 0 if ns >= ew else 1


def apply_action(action):
    traci.trafficlight.setPhase(TLS_ID, action * 2)


def compute_reward(ns, ew, throughput, avg_wait):
    norm_queue      = (ns + ew) / MAX_QUEUE
    norm_wait       = avg_wait / MAX_WAIT
    norm_fairness   = np.std([ns, ew]) / MAX_FAIRNESS
    norm_throughput = throughput / MAX_THROUGHPUT

    queue_score      = max(0.0, 1 - norm_queue)
    wait_score       = max(0.0, 1 - norm_wait)
    fairness_score   = max(0.0, 1 - norm_fairness)
    throughput_score = min(1.0, norm_throughput)

    total = (W_THROUGHPUT * throughput_score + W_QUEUE * queue_score
             + W_WAIT * wait_score + W_FAIRNESS * fairness_score)
    return total


def run_episode(Q, epsilon, seed, learning=True):
    traci.start(["sumo", "-c", SUMOCFG, "--seed", str(seed)])
    build_custom_program()

    step = 0
    prev_state, prev_action = None, None
    total_reward = 0.0

    while step < 3600:
        ns, ew = get_queues()
        state = get_state(ns, ew)

        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = int(np.argmax(Q[state]))

        apply_action(action)
        step += SIGNAL_LEN
        traci.simulationStep(step)

        throughput = traci.simulation.getArrivedNumber()
        avg_wait = get_avg_waiting_time()
        r = compute_reward(ns, ew, throughput, avg_wait)
        total_reward += r

        if learning and prev_state is not None:
            best_next = np.max(Q[state])
            Q[prev_state, prev_action] += ETA * (r + GAMMA * best_next - Q[prev_state, prev_action])

        prev_state, prev_action = state, action

    traci.close()
    return total_reward


if __name__ == "__main__":
    Q = np.zeros((2, 2))

    # ---------------- PHASE 1: TRAINING (multiple episodes) ----------------
    print("=== Phase 1: Training ===")
    with open("training_episode_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "epsilon"])

    epsilon = START_EPSILON
    for ep in range(EPISODES):
        total_r = run_episode(Q, epsilon, seed=ep, learning=True)
        with open("training_episode_summary.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_r, epsilon])
        print(f"Episode {ep:3d} | Reward: {total_r:8.2f} | Epsilon: {epsilon:.3f}")
        epsilon = max(END_EPSILON, epsilon * EPSILON_DECAY)

    print("\nTraining complete.")
    print("Learned Q-table:")
    print(Q)

    # ---------------- PHASE 2: FROZEN EVALUATION ----------------
    print("\n=== Phase 2: Frozen Evaluation (policy fixed, no learning) ===")
    eval_seeds = [101, 102, 103, 104, 105]
    eval_rewards = []
    for seed in eval_seeds:
        r = run_episode(Q, epsilon=0.0, seed=seed, learning=False)
        eval_rewards.append(r)
        print(f"Eval seed {seed} | Reward: {r:.2f}")

    print(f"\nAverage evaluation reward: {np.mean(eval_rewards):.2f}")

    np.save("trained_q_table.npy", Q)
    print("\nSaved trained Q-table to trained_q_table.npy")


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_episode_summary.csv")
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"])
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve: Reward per Episode")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()