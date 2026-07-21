#Code For Finding Out The Reward Function By The All 4 Parameters 



import traci
import numpy as np

TLS_ID = "center"
NS_EDGES = ["N2C", "S2C"]
EW_EDGES = ["E2C", "W2C"]
SIGNAL_LEN = 10

def get_queues():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return ns, ew

def get_avg_waiting_time():
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        return 0.0
    return sum(traci.vehicle.getWaitingTime(v) for v in veh_ids) / len(veh_ids)

def get_phase_label():
    state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
    if "G" in state[:4]:
        return "NS-GREEN", state
    elif "y" in state[:4]:
        return "NS-YELLOW", state
    elif "G" in state[4:]:
        return "EW-GREEN", state
    elif "y" in state[4:]:
        return "EW-YELLOW", state
    else:
        return "ALL-RED", state

def reward_components(ns, ew, throughput, avg_wait,MAX_QUEUE = 9,MAX_THROUGHPUT = 8,MAX_WAIT= 10.3,MAX_FAIRNESS= 4.00):
    queue_penalty     = -(ns + ew)
    throughput_reward = throughput
    wait_penalty       = -avg_wait
    fairness_penalty   = -np.std([ns, ew])
    norm_penalty = queue_penalty/MAX_QUEUE
    norm_throughput = throughput_reward/MAX_THROUGHPUT
    norm_wait= wait_penalty/MAX_WAIT
    norm_fairness= fairness_penalty/MAX_FAIRNESS
    total = norm_throughput + norm_penalty + norm_wait + norm_fairness
    return total, norm_penalty, norm_throughput, norm_wait, norm_fairness

traci.start(["sumo", "-c", "sim.sumocfg"])

step = 0
while step < 3600:   # first 10 minutes is plenty to observe the pattern
    step += SIGNAL_LEN
    traci.simulationStep(step)

    ns, ew = get_queues()
    throughput = traci.simulation.getArrivedNumber()
    avg_wait = get_avg_waiting_time()
    phase_label, raw_state = get_phase_label()

    total, qp, tp_r, wp, fp = reward_components(ns, ew, throughput, avg_wait)

    print(f"t={step:4d}s | phase={phase_label:10s} ({raw_state}) | "
          f"NS={ns} EW={ew} | thrpt={throughput} wait={avg_wait:.1f}s | "
          f"reward={total:6.2f}  [queue={qp:.1f} tp={tp_r:.1f} wait={wp:.1f} fair={fp:.1f}]")

traci.close()