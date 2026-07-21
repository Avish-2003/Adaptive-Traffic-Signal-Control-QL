#Code For Finding Out The Q-Table By Giving User Data (Like: No Of Cars & Traffic Light Duration) & Reward Function By The All 4 Parameters 


import traci
import numpy as np
import subprocess
import matplotlib.pyplot as plt

TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]

# These MUST match your net.net.xml exactly (confirmed earlier via Select-String)
STATE_NS_GREEN  = "GGggrrrrGGggrrrr"
STATE_NS_YELLOW = "yyyyrrrryyyyrrrr"
STATE_EW_GREEN  = "rrrrGGggrrrrGGgg"
STATE_EW_YELLOW = "rrrryyyyrrrryyyy"

# ---------- Step 1: Ask user for configuration ----------
print("=== Simulation Configuration ===")
veh_per_hour = int(input("Vehicles per hour per direction : "))
green_time = int(input("GREEN duration in seconds : "))
yellow_time = int(input("YELLOW duration in seconds : "))
red_time = int(input("RED duration in seconds : "))
print(f"Note: in a 2-phase signal, RED time for one side automatically equals "
      f"the other side's GREEN+YELLOW time ({green_time + yellow_time}s). "
      f"Your RED input ({red_time}s) is shown for reference only.\n")

# ---------- Step 2: Regenerate routes.rou.xml ----------
routes_xml = f"""<routes>
    <vType id="car" length="4.7" minGap="1.3" maxSpeed="10" accel="2.6" decel="4.5"/>
    <route id="NS" edges="N2C C2S"/>
    <route id="SN" edges="S2C C2N"/>
    <route id="EW" edges="E2C C2W"/>
    <route id="WE" edges="W2C C2E"/>
    <flow id="f1" type="car" route="NS" begin="0" end="3600" vehsPerHour="{veh_per_hour}"/>
    <flow id="f2" type="car" route="SN" begin="0" end="3600" vehsPerHour="{veh_per_hour}"/>
    <flow id="f3" type="car" route="EW" begin="0" end="3600" vehsPerHour="{veh_per_hour}"/>
    <flow id="f4" type="car" route="WE" begin="0" end="3600" vehsPerHour="{veh_per_hour}"/>
</routes>
"""
with open("routes.rou.xml", "w") as f:
    f.write(routes_xml)
print(f"routes.rou.xml regenerated with {veh_per_hour} veh/hour per direction.")


# ---------- Step 3: Build the custom TLS program ----------
def build_custom_program():
    phases = [
        traci.trafficlight.Phase(green_time, STATE_NS_GREEN),
        traci.trafficlight.Phase(yellow_time, STATE_NS_YELLOW),
        traci.trafficlight.Phase(green_time, STATE_EW_GREEN),
        traci.trafficlight.Phase(yellow_time, STATE_EW_YELLOW),
    ]
    logic = traci.trafficlight.Logic("custom", 0, 0, phases)
    traci.trafficlight.setProgramLogic(TLS_ID, logic)
    traci.trafficlight.setProgram(TLS_ID, "custom")


# ---------- Step 4: Baseline run (fixed-time, using user's durations) ----------
def run_baseline():
    traci.start(["sumo", "-c", "sim.sumocfg"])
    build_custom_program()

    step = 0
    total_wait, wait_samples = 0.0, 0
    while step < 3600:
        step += 5
        traci.simulationStep(step)
        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            wait_samples += 1
    traci.close()
    return total_wait / wait_samples if wait_samples else 0


# ---------- Step 5: Q-learning run (uses green_time as its decision interval) ----------
def run_qlearning():
    Q = np.zeros((2, 2))
    ETA, GAMMA, EPSILON = 0.1, 0.9, 0.1
    SIGNAL_LEN = green_time

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

    def choose_action(state):
        if np.random.rand() < EPSILON:
            return np.random.choice([0, 1])
        return int(np.argmax(Q[state]))

    def apply_action(action):
        # 0 -> NS green, 1 -> EW green
        traci.trafficlight.setPhase(TLS_ID, action * 2)

    # -------- Reward Function (Normalized) --------
    def reward(ns, ew, throughput, avg_wait,MAX_QUEUE=9, MAX_THROUGHPUT=8, MAX_WAIT=10.3, MAX_FAIRNESS=4.0,W_THROUGHPUT=0.55, W_QUEUE=0.20,W_WAIT=0.15, W_FAIRNESS=0.10):

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
        return total, queue_score, throughput_score, wait_score, fairness_score
    # ---------------- Logging lists ----------------
    time_log = []
    total_log = []
    queue_log = []
    throughput_log = []
    wait_log = []
    fairness_log = []

    # -------- Start Simulation --------
    traci.start(["sumo", "-c", "sim.sumocfg"])
    build_custom_program()

    step = 0
    prev_state = None
    prev_action = None

    total_wait = 0.0
    wait_samples = 0

    while step < 3600:

        ns, ew = get_queues()
        state = get_state(ns, ew)
        action = choose_action(state)
        apply_action(action)

        step += SIGNAL_LEN
        traci.simulationStep(step)

        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            wait_samples += 1

        throughput = traci.simulation.getArrivedNumber()
        avg_wait = get_avg_waiting_time()

        (
            r,
            q_pen,
            t_reward,
            w_pen,
            f_pen
        ) = reward(
            ns,
            ew,
            throughput,
            avg_wait
        )

        # Optional: Print reward components
        print(
            f"Step {step:4d} | "
            f"Queue={q_pen:.3f} "
            f"Throughput={t_reward:.3f} "
            f"Wait={w_pen:.3f} "
            f"Fairness={f_pen:.3f} "
            f"Total={r:.3f}"
        )

        # Record every step's values for plotting
        time_log.append(step)
        total_log.append(r)
        queue_log.append(q_pen)
        throughput_log.append(t_reward)
        wait_log.append(w_pen)
        fairness_log.append(f_pen)

        if prev_state is not None:
            best_next = np.max(Q[state])
            Q[prev_state, prev_action] += ETA * (
                r + GAMMA * best_next - Q[prev_state, prev_action]
            )

        prev_state = state
        prev_action = action

    traci.close()

    avg_wait_overall = (
        total_wait / wait_samples
        if wait_samples
        else 0
    )

    return Q, avg_wait_overall, time_log, total_log, queue_log, throughput_log, wait_log, fairness_log


# ---------- Run both ----------
print("Running FIXED-TIME baseline...")
baseline_wait = run_baseline()

print("Running Q-LEARNING controller...")
(
    Q_table,
    ql_wait,
    time_log,
    total_log,
    queue_log,
    throughput_log,
    wait_log,
    fairness_log,
) = run_qlearning()

# ---------- Results ----------
print("\n================ RESULTS ================")
print(f"Vehicles/hour per direction : {veh_per_hour}")
print(f"Green / Yellow durations    : {green_time}s / {yellow_time}s")
print(f"Fixed-Time avg waiting time : {baseline_wait:.2f} s")
print(f"Q-Learning avg waiting time : {ql_wait:.2f} s")
if baseline_wait > 0:
    improvement = (baseline_wait - ql_wait) / baseline_wait * 100
    print(f"Improvement                 : {improvement:.1f}%")
print("\nFinal learned Q-table:")
print(Q_table)


# =========================================================
# CHART 1: LINE CHART — how each parameter evolves over time
# =========================================================
# ---------------- Smoothing helper ----------------
def smooth(data, window=15):
    data = np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

window = 15
t_smooth = time_log[window-1:]

# =========================================================
# CHART 1 (Option A): Smoothed, single overlapping chart
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(t_smooth, smooth(total_log, window), label="Total Reward", color="black", linewidth=2.5)
plt.plot(t_smooth, smooth(queue_log, window), label="Queue Score", linewidth=1.8)
plt.plot(t_smooth, smooth(throughput_log, window), label="Throughput Score", linewidth=1.8)
plt.plot(t_smooth, smooth(wait_log, window), label="Wait Score", linewidth=1.8)
plt.plot(t_smooth, smooth(fairness_log, window), label="Fairness Score", linewidth=1.8)
plt.xlabel("Simulation Time (seconds)")
plt.ylabel("Reward Contribution (0 to 1 scale, smoothed)")
plt.title("Effect of Each Parameter on the Reward Function Over Time (Smoothed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reward_components_line.png", dpi=150)
plt.show()

# =========================================================
# CHART 1 (Option B): Separated subplots — cleanest for presentations
# =========================================================
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
series = [
    ("Total Reward", total_log, "black"),
    ("Queue Score", queue_log, "tomato"),
    ("Throughput Score", throughput_log, "seagreen"),
    ("Wait Score", wait_log, "orange"),
    ("Fairness Score", fairness_log, "steelblue"),
]
for ax, (label, data, color) in zip(axes, series):
    ax.plot(t_smooth, smooth(data, window), color=color, linewidth=2)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Simulation Time (seconds)")
fig.suptitle("Reward Components Over Time (Separated View)", fontsize=13)
plt.tight_layout()
plt.savefig("reward_components_subplots.png", dpi=150)
plt.show()


# =========================================================
# CHART 2: BAR CHART — average contribution per parameter (whole run)
# =========================================================
avg_queue = np.mean(queue_log)
avg_throughput = np.mean(throughput_log)
avg_wait = np.mean(wait_log)
avg_fairness = np.mean(fairness_log)

labels = ["Queue Penalty", "Throughput Reward", "Wait Penalty", "Fairness Penalty"]
values = [avg_queue, avg_throughput, avg_wait, avg_fairness]
colors = ["tomato", "seagreen", "orange", "steelblue"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Average Contribution (normalized)")
plt.title("Average Effect of Each Parameter on Reward (Whole Simulation)")

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.3f}",
              ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

plt.tight_layout()
plt.savefig("reward_components_bar_avg.png", dpi=150)
plt.show()
print("Average bar chart saved as reward_components_bar_avg.png")



labels = ["Fixed-Time", "Q-Learning"]
values = [baseline_wait, ql_wait]
colors = ["gray", "seagreen"]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=12)
plt.ylabel("Average Waiting Time (s)")
plt.title(f"Fixed-Time vs Q-Learning ({veh_per_hour} veh/hr)")
plt.tight_layout()
plt.savefig("baseline_vs_qlearning.png", dpi=150)
plt.show()





fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(Q_table, cmap="RdYlGn")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Action: NS-Green", "Action: EW-Green"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["State: NS≥EW", "State: EW>NS"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{Q_table[i,j]:.2f}", ha="center", va="center", color="black", fontsize=12)
plt.colorbar(im, label="Q-value")
plt.title("Learned Q-Table (Green = Preferred Action)")
plt.tight_layout()
plt.savefig("qtable_heatmap.png", dpi=150)
plt.show()