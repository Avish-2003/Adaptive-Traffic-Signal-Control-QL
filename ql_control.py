#Code For Finding Out The Q-Table & Reward Function By The All 2 Parameters (Standard Deviation Of Queue Length & Throughput)
#It Is The Begining Code
#But It has a Major Fault In Reward Function (avg_waiting_time can't be add with other numbers need normalization)

import traci
import numpy as np

ETA = 0.1
GAMMA = 0.9
EPSILON = 0.1
DELTA = 0.5
SIGNAL_LEN = 10

TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]

Q = np.zeros((2, 2))

def get_queues():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return ns, ew

def get_state(ns, ew):
    return 0 if ns >= ew else 1

def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.choice([0, 1])
    return int(np.argmax(Q[state]))

def apply_action(action):
    traci.trafficlight.setPhase(TLS_ID, action * 2)

def compute_reward(ns, ew, throughput):
    dql = np.std([ns, ew])
    tau_tp = np.exp(-throughput / 20.0)
    alpha = 0.5
    f_t = alpha * dql + (1 - alpha) * tau_tp
    f_t = max(f_t, 1e-6)
    return np.log(f_t) / np.log(DELTA)

traci.start(["sumo", "-c", "sim.sumocfg"])

step = 0
prev_state, prev_action = None, None
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
    r = compute_reward(ns, ew, throughput)

    if prev_state is not None:
        best_next = np.max(Q[state])
        Q[prev_state, prev_action] += ETA * (r + GAMMA * best_next - Q[prev_state, prev_action])

    prev_state, prev_action = state, action
    print(f"t={step}s | NS={ns} EW={ew} | action={action} | reward={r:.3f}")

traci.close()

avg_wait = total_wait / wait_samples if wait_samples else 0
print("\nFinal learned Q-table:")
print(Q)
print(f"\n[Q-LEARNING] Average waiting time per vehicle-step: {avg_wait:.2f} seconds")