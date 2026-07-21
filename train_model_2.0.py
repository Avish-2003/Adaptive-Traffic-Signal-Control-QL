import traci
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]

STATE_NS_GREEN  = "GGggrrrrGGggrrrr"
STATE_NS_YELLOW = "yyyyrrrryyyyrrrr"
STATE_EW_GREEN  = "rrrrGGggrrrrGGgg"
STATE_EW_YELLOW = "rrrryyyyrrrryyyy"

ETA, GAMMA = 0.1, 0.9
GREEN_TIME, YELLOW_TIME, SIGNAL_LEN = 10, 3, 10   # FIXED — same for every test
SUMOCFG = "sim.sumocfg"

W_THROUGHPUT, W_QUEUE, W_WAIT, W_FAIRNESS = 0.55, 0.20, 0.15, 0.10
MAX_QUEUE, MAX_THROUGHPUT, MAX_WAIT, MAX_FAIRNESS = 9, 8, 10.3, 4.0

# ---------------- Training volumes (agent learns from these) ----------------
TRAIN_VOLUMES = [200, 400, 600, 800]
EPISODES_PER_VOLUME = 15   # 4 volumes x 15 = 60 total training episodes

# ---------------- Testing volumes (held out — never trained on) ----------------
TEST_VOLUMES = [1000, 1200]
EVAL_SEEDS_PER_VOLUME = 5


def write_routes(veh_per_hour):
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
</routes>"""
    with open("routes.rou.xml", "w") as f:
        f.write(routes_xml)


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


# ---- FIX: now returns all 5 values (total + the 4 individual scores) ----
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
    return total, queue_score, throughput_score, wait_score, fairness_score


def run_episode(Q, epsilon, seed, veh_per_hour, learning=True):
    write_routes(veh_per_hour)
    traci.start(["sumo", "-c", SUMOCFG, "--seed", str(seed)])
    build_custom_program()

    step = 0
    prev_state, prev_action = None, None
    total_reward = 0.0
    total_wait, wait_samples = 0.0, 0

    # accumulators for the 4 component scores, so we can log per-episode averages
    sum_queue_score, sum_throughput_score, sum_wait_score, sum_fairness_score = 0.0, 0.0, 0.0, 0.0
    n_steps = 0

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

        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            wait_samples += 1

        throughput = traci.simulation.getArrivedNumber()
        avg_wait = get_avg_waiting_time()

        # ---- FIX: correctly unpack all 5 returned values ----
        r, q_score, t_score, w_score, f_score = compute_reward(ns, ew, throughput, avg_wait)
        total_reward += r

        sum_queue_score += q_score
        sum_throughput_score += t_score
        sum_wait_score += w_score
        sum_fairness_score += f_score
        n_steps += 1

        if learning and prev_state is not None:
            best_next = np.max(Q[state])
            Q[prev_state, prev_action] += ETA * (r + GAMMA * best_next - Q[prev_state, prev_action])

        prev_state, prev_action = state, action

    traci.close()
    avg_wait_overall = total_wait / wait_samples if wait_samples else 0

    # per-episode average scores (what the parameter-vs-volume charts need)
    avg_queue_score = sum_queue_score / n_steps if n_steps else 0
    avg_throughput_score = sum_throughput_score / n_steps if n_steps else 0
    avg_wait_score = sum_wait_score / n_steps if n_steps else 0
    avg_fairness_score = sum_fairness_score / n_steps if n_steps else 0

    return (total_reward, avg_wait_overall,
            avg_queue_score, avg_throughput_score, avg_wait_score, avg_fairness_score)


if __name__ == "__main__":
    Q = np.zeros((2, 2))

    # ============ PHASE 1: TRAINING across multiple volumes ============
    print(f"=== Phase 1: Training (Green={GREEN_TIME}s, Yellow={YELLOW_TIME}s fixed) ===")
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # FIX: added global_episode + the 4 score columns the charts need
        writer.writerow(["volume", "episode", "global_episode", "seed", "total_reward",
                          "avg_wait", "epsilon", "queue_score", "throughput_score",
                          "wait_score", "fairness_score"])

    epsilon = 1.0
    seed_counter = 0
    global_ep = 0
    for volume in TRAIN_VOLUMES:
        for ep in range(EPISODES_PER_VOLUME):
            (r, avg_wait, q_score, t_score, w_score, f_score) = run_episode(
                Q, epsilon, seed=seed_counter, veh_per_hour=volume, learning=True
            )
            with open("training_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([volume, ep, global_ep, seed_counter, r, avg_wait,
                                  epsilon, q_score, t_score, w_score, f_score])
            print(f"Volume={volume:4d} veh/hr | Ep {ep:2d} | Reward={r:7.2f} | "
                  f"AvgWait={avg_wait:.2f}s | Eps={epsilon:.3f}")
            epsilon = max(0.05, epsilon * 0.94)
            seed_counter += 1
            global_ep += 1

    print("\nTraining complete. Learned Q-table:")
    print(Q)

    # ============ PHASE 2: TESTING on held-out volumes (never trained on) ============
    print(f"\n=== Phase 2: Testing on HELD-OUT volumes {TEST_VOLUMES} ===")
    with open("testing_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["volume", "eval_run", "seed", "total_reward", "avg_wait",
                          "queue_score", "throughput_score", "wait_score", "fairness_score"])

    for volume in TEST_VOLUMES:
        for i in range(EVAL_SEEDS_PER_VOLUME):
            seed = 9000 + i  # distinct seed range, guaranteed not used in training
            (r, avg_wait, q_score, t_score, w_score, f_score) = run_episode(
                Q, epsilon=0.0, seed=seed, veh_per_hour=volume, learning=False
            )
            with open("testing_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([volume, i, seed, r, avg_wait, q_score, t_score, w_score, f_score])
            print(f"[TEST] Volume={volume:4d} veh/hr | Run {i} | Reward={r:7.2f} | AvgWait={avg_wait:.2f}s")

    np.save("trained_q_table.npy", Q)
    print("\nSaved trained_q_table.npy")

    # =========================================================================
    # PLOTTING — everything below reads the CSVs we just wrote
    # =========================================================================
    df = pd.read_csv("training_log.csv")
    test_df = pd.read_csv("testing_log.csv")

    def smooth(data, window=5):
        data = np.array(data, dtype=float)
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid')

    window = 5
    ge_smooth = df["global_episode"].values[window - 1:]

    # ---------------- Chart A: Smoothed learning curve + epsilon decay ----------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["global_episode"], df["total_reward"], color="lightgray", linewidth=1, label="Raw Reward")
    ax1.plot(ge_smooth, smooth(df["total_reward"], window), color="black", linewidth=2.5, label="Smoothed Reward")
    ax1.set_xlabel("Training Episode (across all volumes)")
    ax1.set_ylabel("Total Reward per Episode")
    ax1.grid(alpha=0.3)

    volumes = df["volume"].unique()
    episodes_per_volume = len(df) // len(volumes)
    for i, v in enumerate(volumes):
        boundary = i * episodes_per_volume
        ax1.axvline(boundary, color="steelblue", linestyle=":", alpha=0.6)
        ax1.text(boundary + 1, ax1.get_ylim()[1] * 0.97, f"{v} veh/hr", fontsize=9, color="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(df["global_episode"], df["epsilon"], color="tomato", linewidth=1.8, linestyle="--", label="Epsilon")
    ax2.set_ylabel("Epsilon", color="tomato")
    ax2.tick_params(axis='y', labelcolor="tomato")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.title("Training Progress: Reward Growth vs. Epsilon Decay (Across Traffic Volumes)")
    plt.tight_layout()
    plt.savefig("learning_curve_smoothed.png", dpi=150)
    plt.show()

    # ---------------- Chart B & C: Bar charts by volume ----------------
    for metric, ylabel, title, fname in [
        ("avg_wait", "Average Waiting Time (s)", "Average Waiting Time by Training Traffic Volume", "avg_wait_by_volume_bar.png"),
        ("total_reward", "Average Total Reward per Episode", "Average Training Reward by Traffic Volume", "avg_reward_by_volume_bar.png"),
    ]:
        avg_by_volume = df.groupby("volume")[metric].mean()
        plt.figure(figsize=(8, 5))
        bars = plt.bar([str(v) for v in avg_by_volume.index], avg_by_volume.values,
                       color=["#8fd694", "#6fc2e0", "#f2b84b", "#e17a7a"])
        for bar, val in zip(bars, avg_by_volume.values):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=11)
        plt.xlabel("Traffic Volume (vehicles/hour per direction)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.show()

    # ---------------- Chart D: Small multiples, one subplot per volume ----------------
    fig, axes = plt.subplots(1, len(volumes), figsize=(16, 4), sharey=True)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    for ax, v, color in zip(axes, volumes, colors):
        sub = df[df["volume"] == v].reset_index(drop=True)
        ax.plot(sub["episode"], sub["total_reward"], color=color, marker='o', markersize=3, linewidth=1.8)
        ax.set_title(f"{v} veh/hr")
        ax.set_xlabel("Episode (within volume)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Total Reward")
    fig.suptitle("Reward Progression Within Each Training Volume", fontsize=13)
    plt.tight_layout()
    plt.savefig("reward_per_volume_smallmultiples.png", dpi=150)
    plt.show()

    # ---------------- Chart E: Parameters vs Volume (grouped bar) ----------------
    avg_by_volume_params = df.groupby("volume")[
        ["queue_score", "throughput_score", "wait_score", "fairness_score"]
    ].mean()
    param_volumes = avg_by_volume_params.index.tolist()
    labels = ["Queue Score", "Throughput Score", "Wait Score", "Fairness Score"]
    param_colors = ["tomato", "seagreen", "orange", "steelblue"]

    x = np.arange(len(param_volumes))
    width = 0.2
    plt.figure(figsize=(11, 6))
    for i, (col, label, color) in enumerate(zip(
            ["queue_score", "throughput_score", "wait_score", "fairness_score"], labels, param_colors)):
        offset = (i - 1.5) * width
        bars = plt.bar(x + offset, avg_by_volume_params[col], width, label=label, color=color)
        for bar, val in zip(bars, avg_by_volume_params[col]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, [f"{v} veh/hr" for v in param_volumes])
    plt.ylabel("Average Score (0 to 1 scale)")
    plt.title("Effect of Traffic Volume on Each Reward Parameter")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("params_vs_volume_bar.png", dpi=150)
    plt.show()

    # ---------------- Chart F: Train vs Test trend continuity ----------------
    train_avg_wait = df.groupby("volume")["avg_wait"].mean()
    test_avg_wait = test_df.groupby("volume")["avg_wait"].mean()
    train_avg_reward = df.groupby("volume")["total_reward"].mean()
    test_avg_reward = test_df.groupby("volume")["total_reward"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, train_series, test_series, ylabel, title in [
        (axes[0], train_avg_wait, test_avg_wait, "Average Waiting Time (s)", "Waiting Time: Training vs Held-Out Testing"),
        (axes[1], train_avg_reward, test_avg_reward, "Average Total Reward per Episode", "Reward: Training vs Held-Out Testing"),
    ]:
        ax.plot(train_series.index, train_series.values, marker='o', color="steelblue",
                linewidth=2.2, markersize=8, label="Training volumes")
        ax.plot(test_series.index, test_series.values, marker='s', color="tomato",
                linewidth=2.2, markersize=9, linestyle="--", label="Testing volumes (held-out)")
        ax.plot([train_series.index[-1], test_series.index[0]],
                [train_series.values[-1], test_series.values[0]],
                color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Traffic Volume (veh/hr)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("train_vs_test_trend.png", dpi=150)
    plt.show()

    # ---------------- Chart G: Overall Train vs Test bar comparison ----------------
    metrics = {
        "Avg Waiting Time (s)": (df["avg_wait"].mean(), test_df["avg_wait"].mean()),
        "Avg Reward /Episode": (df["total_reward"].mean(), test_df["total_reward"].mean()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (metric, (train_val, test_val)) in zip(axes, metrics.items()):
        bars = ax.bar(["Training", "Testing (held-out)"], [train_val, test_val],
                      color=["steelblue", "tomato"], width=0.5)
        for bar, val in zip(bars, [train_val, test_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(metric)
    fig.suptitle("Overall Comparison: Training Performance vs Testing (Generalization) Performance", fontsize=13)
    plt.tight_layout()
    plt.savefig("train_vs_test_overall_bar.png", dpi=150)
    plt.show()
