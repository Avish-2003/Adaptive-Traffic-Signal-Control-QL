"""
General-purpose SUMO traffic signal simulator + Q-learning controller.

Works on ANY junction (3-way, 4-way, 5-way, 6-way, multiple lights, etc.)
No hardcoded edge names or phase state strings — everything is discovered
at runtime directly from the loaded network.

User provides: vehicles/hour, green duration, yellow duration, red duration (informational).
Runs a FIXED-TIME baseline and a Q-LEARNING controller back-to-back and compares them.


N.B.:- It Is The Corrected Version Of ql_control.py & Generalize Version Of configure_run.py & reward_observer.py
But Instead Of Showing All The Parameter Value It Only Calculate The Actions ACcording To State By Given Simulation And User Input Data.
But To Run This Code We Need To Input Some Custom Value Like Max Values Of All The Parameters For Normalize Value Which Always Changes According To The Simulation & For That We Also Need The max_min.py
"""

import traci
import numpy as np
import xml.etree.ElementTree as ET

# ---------------- FILE PATHS (edit if your files are named differently) ----------------
SUMOCFG = "sim.sumocfg" #Change The File Name
ROUTES_FILE = "routes.rou.xml" #Change The File Name
SIM_DURATION = 3600   # seconds
ETA = 0.1
GAMMA = 0.9
EPSILON = 0.1
# -----------------------------------------------------------------------------------------


# ============================================================
# STEP 1: USER CONFIGURATION
# ============================================================
def get_user_config():
    print("=== Simulation Configuration ===")
    veh_per_hour = int(input("Vehicles per hour per direction (e.g. 300, 500, 1000): "))
    green_time = int(input("GREEN duration in seconds (e.g. 10): "))
    yellow_time = int(input("YELLOW duration in seconds (e.g. 3): "))
    red_time = int(input("RED duration in seconds (informational only, e.g. 20): "))
    print(f"Note: RED time for any one direction is automatically however long the OTHER "
          f"green phases + yellow phases take. Your RED input ({red_time}s) is for reference only.\n")
    return veh_per_hour, green_time, yellow_time


# ============================================================
# STEP 2: SCALE TRAFFIC VOLUME (generic — works for any route file)
# ============================================================
def update_routes_file(veh_per_hour):
    tree = ET.parse(ROUTES_FILE)
    root = tree.getroot()
    count = 0
    for flow in root.findall("flow"):
        flow.set("vehsPerHour", str(veh_per_hour))
        count += 1
    tree.write(ROUTES_FILE)
    print(f"[INFO] Updated {count} <flow> entries in {ROUTES_FILE} to {veh_per_hour} veh/hour.")


# ============================================================
# STEP 3: JUNCTION DISCOVERY (generic — works for any network)
# ============================================================
def discover_traffic_light():
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        raise RuntimeError("No traffic lights found in this network.")
    tls_id = tls_ids[0]
    print(f"[INFO] Using traffic light: {tls_id}")
    return tls_id


def discover_phases(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.phases
    green_indices = [i for i, p in enumerate(phases) if ('G' in p.state or 'g' in p.state) and 'y' not in p.state]
    print(f"[INFO] Found {len(phases)} total phases, {len(green_indices)} green phases: {green_indices}")
    for i, p in enumerate(phases):
        tag = "GREEN" if i in green_indices else ("YELLOW" if 'y' in p.state else "OTHER")
        print(f"        Phase {i} [{tag}]: state='{p.state}'")
    return phases, green_indices


def map_phase_to_lanes(tls_id, phases, green_indices):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    phase_lanes = {}
    for gp in green_indices:
        state = phases[gp].state
        lanes = set()
        for i, link_group in enumerate(controlled_links):
            if i < len(state) and state[i] in ('G', 'g'):
                for link in link_group:
                    lanes.add(link[0])   # incoming lane
        phase_lanes[gp] = lanes
        print(f"[INFO] Phase {gp} controls lanes: {lanes}")
    return phase_lanes


def build_custom_program(tls_id, phases, green_indices, green_time, yellow_time):
    """Rebuilds the junction's phase program using the user's timing, keeping the
       original (auto-discovered) state strings — works for any number of phases."""
    new_phases = []
    for i, p in enumerate(phases):
        if i in green_indices:
            duration = green_time
        elif 'y' in p.state:
            duration = yellow_time
        else:
            duration = p.duration  # leave all-red/other phases as-is
        new_phases.append(traci.trafficlight.Phase(duration, p.state))
    logic = traci.trafficlight.Logic("custom", 0, 0, new_phases)
    traci.trafficlight.setProgramLogic(tls_id, logic)
    traci.trafficlight.setProgram(tls_id, "custom")


# ============================================================
# STEP 4: MEASUREMENT HELPERS (generic)
# ============================================================
def get_queue_per_phase(phase_lanes, green_indices):
    return [sum(traci.lane.getLastStepHaltingNumber(l) for l in phase_lanes[gp]) for gp in green_indices]


def get_avg_waiting_time():
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        return 0.0
    return sum(traci.vehicle.getWaitingTime(v) for v in veh_ids) / len(veh_ids)


# ============================================================
# STEP 5: REWARD FUNCTION (generic — works for any number of directions)
# ============================================================
def compute_reward(queues, throughput, avg_wait, MAX_QUEUE=9, MAX_THROUGHPUT=8, MAX_WAIT=10.3, MAX_FAIRNESS=4.0):
    queue_penalty     = -sum(queues)
    throughput_reward = throughput
    wait_penalty       = -avg_wait
    fairness_penalty   = -np.std(queues)

    norm_queue      = queue_penalty / MAX_QUEUE
    norm_throughput = throughput_reward / MAX_THROUGHPUT
    norm_wait       = wait_penalty / MAX_WAIT
    norm_fairness   = fairness_penalty / MAX_FAIRNESS

    total = norm_throughput + norm_queue + norm_wait + norm_fairness
    return total, norm_queue, norm_throughput, norm_wait, norm_fairness


# ============================================================
# STEP 6: FIXED-TIME BASELINE RUN
# ============================================================
def run_baseline(green_time, yellow_time):
    traci.start(["sumo", "-c", SUMOCFG])
    tls_id = discover_traffic_light()
    phases, green_indices = discover_phases(tls_id)
    build_custom_program(tls_id, phases, green_indices, green_time, yellow_time)

    step = 0
    total_wait, wait_samples = 0.0, 0
    while step < SIM_DURATION:
        step += 5
        traci.simulationStep(step)
        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            wait_samples += 1
    traci.close()
    return total_wait / wait_samples if wait_samples else 0


# ============================================================
# STEP 7: Q-LEARNING RUN (generic — any number of phases/directions)
# ============================================================
def run_qlearning(green_time):
    traci.start(["sumo", "-c", SUMOCFG])
    tls_id = discover_traffic_light()
    phases, green_indices = discover_phases(tls_id)
    phase_lanes = map_phase_to_lanes(tls_id, phases, green_indices)

    num_actions = len(green_indices)
    Q = np.zeros((num_actions, num_actions))
    SIGNAL_LEN = green_time

    def get_state(queues):
        return int(np.argmax(queues))

    def choose_action(state):
        if np.random.rand() < EPSILON:
            return np.random.randint(num_actions)
        return int(np.argmax(Q[state]))

    def apply_action(action):
        traci.trafficlight.setPhase(tls_id, green_indices[action])

    step = 0
    prev_state, prev_action = None, None
    total_wait, wait_samples = 0.0, 0

    while step < SIM_DURATION:
        queues = get_queue_per_phase(phase_lanes, green_indices)
        state = get_state(queues)
        action = choose_action(state)
        apply_action(action)

        step += SIGNAL_LEN
        traci.simulationStep(step)

        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            wait_samples += 1

        throughput = traci.simulation.getArrivedNumber()
        avg_wait = get_avg_waiting_time()
        r, q_pen, tp_reward, w_pen, f_pen = compute_reward(queues, throughput, avg_wait)

        if prev_state is not None:
            best_next = np.max(Q[state])
            Q[prev_state, prev_action] += ETA * (r + GAMMA * best_next - Q[prev_state, prev_action])

        prev_state, prev_action = state, action

    traci.close()
    avg_wait_overall = total_wait / wait_samples if wait_samples else 0
    return Q, avg_wait_overall, num_actions


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    veh_per_hour, green_time, yellow_time = get_user_config()
    update_routes_file(veh_per_hour)

    print("\nRunning FIXED-TIME baseline...")
    baseline_wait = run_baseline(green_time, yellow_time)

    print("\nRunning Q-LEARNING controller...")
    Q_table, ql_wait, num_actions = run_qlearning(green_time)

    print("\n================ RESULTS ================")
    print(f"Vehicles/hour per direction : {veh_per_hour}")
    print(f"Green / Yellow durations     : {green_time}s / {yellow_time}s")
    print(f"Number of directions/actions : {num_actions}")
    print(f"Fixed-Time avg waiting time  : {baseline_wait:.2f} s")
    print(f"Q-Learning avg waiting time  : {ql_wait:.2f} s")
    if baseline_wait > 0:
        improvement = (baseline_wait - ql_wait) / baseline_wait * 100
        print(f"Improvement                   : {improvement:.1f}%")
    print("\nFinal learned Q-table:")
    print(Q_table)