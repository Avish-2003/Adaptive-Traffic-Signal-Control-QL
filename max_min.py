#This Code Is For Finding Out The Maximum Values Of All Parameters To Find The Normalize Value.


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

traci.start(["sumo", "-c", "sim.sumocfg"])

max_queue_total = 0
max_throughput = 0
max_wait = 0
max_fairness = 0

step = 0
while step < 3600:
    step += SIGNAL_LEN
    traci.simulationStep(step)

    ns, ew = get_queues()
    throughput = traci.simulation.getArrivedNumber()
    avg_wait = get_avg_waiting_time()
    fairness = np.std([ns, ew])

    max_queue_total = max(max_queue_total, ns + ew)
    max_throughput = max(max_throughput, throughput)
    max_wait = max(max_wait, avg_wait)
    max_fairness = max(max_fairness, fairness)

traci.close()

print("=== Observed Maximums (use these to normalize) ===")
print(f"MAX_QUEUE      = {max_queue_total}")
print(f"MAX_THROUGHPUT = {max_throughput}")
print(f"MAX_WAIT       = {max_wait:.1f}")
print(f"MAX_FAIRNESS   = {max_fairness:.2f}")