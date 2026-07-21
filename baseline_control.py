#This Code Is For Finding Out The Fixed Waiting Time To Compare With The QL Waiting Time.


import traci

traci.start(["sumo", "-c", "sim.sumocfg"])

step = 0
total_wait = 0.0
wait_samples = 0

while step < 3600:
    step += 10
    traci.simulationStep(step)

    for veh_id in traci.vehicle.getIDList():
        total_wait += traci.vehicle.getWaitingTime(veh_id)
        wait_samples += 1

traci.close()

avg_wait = total_wait / wait_samples if wait_samples else 0
print(f"\n[BASELINE - Fixed Timer] Average waiting time per vehicle-step: {avg_wait:.2f} seconds")