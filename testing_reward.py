#This Chunk Of Code Use To Test The Custom Simulated Value In Term Of CSV Or Excel file But It Will Give Only Staticstical Analysis



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load the dataset ----------------
df = pd.read_csv(r"C:\Users\Avisanta Saha\Downloads\metrics_dur_10s_cars_1200total_1000s.csv")

print("Columns found:", list(df.columns))
print(f"Total rows (time steps): {len(df)}")
print(df.head())

# ---------------- Reward function (same as your existing one) ----------------
MAX_QUEUE = 9
MAX_THROUGHPUT = 8
MAX_WAIT = 10.3
MAX_FAIRNESS = 4.0

def compute_reward(ns, ew, throughput, avg_wait):
    queue_penalty     = -(ns + ew)
    throughput_reward = throughput
    wait_penalty       = -avg_wait
    fairness_penalty   = -np.std([ns, ew])

    norm_queue      = queue_penalty / MAX_QUEUE
    norm_throughput = throughput_reward / MAX_THROUGHPUT
    norm_wait       = wait_penalty / MAX_WAIT
    norm_fairness   = fairness_penalty / MAX_FAIRNESS

    total = norm_queue + norm_throughput + norm_wait + norm_fairness
    return total, norm_queue, norm_throughput, norm_wait, norm_fairness


# ---------------- Apply reward function to every row ----------------
total_log, queue_log, throughput_log, wait_log, fairness_log = [], [], [], [], []

for _, row in df.iterrows():
    ns = row["Queue_North_South"]
    ew = row["Queue_West_East"]
    throughput = row["Throughput_Step"]
    # Combine both directions' waiting time into a single average for this row
    avg_wait = (row["Avg_Waiting_Time_West_East"] + row["Avg_Waiting_Time_North_South"]) / 2

    total, qp, tp, wp, fp = compute_reward(ns, ew, throughput, avg_wait)

    total_log.append(total)
    queue_log.append(qp)
    throughput_log.append(tp)
    wait_log.append(wp)
    fairness_log.append(fp)

df["Reward_Total"] = total_log
df["Reward_Queue"] = queue_log
df["Reward_Throughput"] = throughput_log
df["Reward_Wait"] = wait_log
df["Reward_Fairness"] = fairness_log

print("\nReward statistics:")
print(df[["Reward_Total", "Reward_Queue", "Reward_Throughput", "Reward_Wait", "Reward_Fairness"]].describe())

# Save the full result (original data + computed rewards) as a new CSV
df.to_csv("metrics_with_rewards.csv", index=False)
print("\nSaved: metrics_with_rewards.csv")

# ---------------- Plot: reward over time ----------------
plt.figure(figsize=(14, 6))
plt.plot(df["Step"], df["Reward_Total"], color="black", linewidth=1.5, label="Total Reward")
plt.plot(df["Step"], df["Reward_Queue"], linestyle="--", alpha=0.7, label="Queue Penalty")
plt.plot(df["Step"], df["Reward_Throughput"], linestyle="--", alpha=0.7, label="Throughput Reward")
plt.plot(df["Step"], df["Reward_Wait"], linestyle="--", alpha=0.7, label="Wait Penalty")
plt.plot(df["Step"], df["Reward_Fairness"], linestyle="--", alpha=0.7, label="Fairness Penalty")
plt.xlabel("Simulation Step")
plt.ylabel("Reward Contribution (normalized)")
plt.title("Reward Function Applied to Mentor-Provided Dataset (1000s, 1200 vehicles)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mentor_dataset_reward_plot.png", dpi=150)
plt.show()