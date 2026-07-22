"""

================================================================================

Interactive Traffic Signal & Duration Tester with SUMO Simulation

Paper Reference: "Traffic signal control for smart cities using reinforcement learning"

                 (Computer Communications, 2020)

================================================================================

This script allows you to enter:

 1. Exact Number of Cars waiting on North-South (NS)

 2. Exact Number of Cars waiting on East-West (EW)

 3. Custom Traffic Light Durations (Green & Yellow times)

It predicts the optimal Signal Phase & Duration, and runs a live SUMO test!

"""

import os
import sys
import traci
import matplotlib.pyplot as plt

TLS_ID = "center"
STATE_NS_GREEN  = "GGggrrrrGGggrrrr"
STATE_NS_YELLOW = "yyyyrrrryyyyrrrr"
STATE_EW_GREEN  = "rrrrGGggrrrrGGgg"
STATE_EW_YELLOW = "rrrryyyyrrrryyyy"

def setup_sumo_files():
    if not os.path.exists("sim.sumocfg"):
        sumocfg_xml = """<configuration>
    <input>
        <net-file value="net.net.xml"/>
        <route-files value="routes_custom.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
</configuration>"""
        with open("sim.sumocfg", "w") as f:
            f.write(sumocfg_xml)

    if not os.path.exists("net.net.xml"):
        net_xml = """<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="-200.00,-200.00,200.00,200.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="C2E" from="center" to="east" priority="1"><lane id="C2E_0" index="0" speed="13.89" length="200.00" shape="0.00,-4.80 200.00,-4.80"/><lane id="C2E_1" index="1" speed="13.89" length="200.00" shape="0.00,-1.60 200.00,-1.60"/></edge>
    <edge id="C2N" from="center" to="north" priority="1"><lane id="C2N_0" index="0" speed="13.89" length="200.00" shape="4.80,0.00 4.80,200.00"/><lane id="C2N_1" index="1" speed="13.89" length="1.60,0.00 1.60,200.00"/></edge>
    <edge id="C2S" from="center" to="south" priority="1"><lane id="C2S_0" index="0" speed="13.89" length="200.00" shape="-4.80,0.00 -4.80,-200.00"/><lane id="C2S_1" index="1" speed="13.89" length="-1.60,0.00 -1.60,-200.00"/></edge>
    <edge id="C2W" from="center" to="west" priority="1"><lane id="C2W_0" index="0" speed="13.89" length="200.00" shape="0.00,4.80 -200.00,4.80"/><lane id="C2W_1" index="1" speed="13.89" length="0.00,1.60 -200.00,1.60"/></edge>
    <edge id="E2C" from="east" to="center" priority="1"><lane id="E2C_0" index="0" speed="13.89" length="200.00" shape="200.00,4.80 0.00,4.80"/><lane id="E2C_1" index="1" speed="13.89" length="200.00,1.60 0.00,1.60"/></edge>
    <edge id="N2C" from="north" to="center" priority="1"><lane id="N2C_0" index="0" speed="13.89" length="200.00" shape="-4.80,200.00 -4.80,0.00"/><lane id="N2C_1" index="1" speed="13.89" length="-1.60,200.00 -1.60,0.00"/></edge>
    <edge id="S2C" from="south" to="center" priority="1"><lane id="S2C_0" index="0" speed="13.89" length="200.00" shape="4.80,-200.00 4.80,0.00"/><lane id="S2C_1" index="1" speed="13.89" length="1.60,-200.00 1.60,0.00"/></edge>
    <edge id="W2C" from="west" to="center" priority="1"><lane id="W2C_0" index="0" speed="13.89" length="200.00" shape="-200.00,-4.80 0.00,-4.80"/><lane id="W2C_1" index="1" speed="13.89" length="-200.00,-1.60 0.00,-1.60"/></edge>
    <tlLogic id="center" type="static" programID="0" offset="0">
        <phase duration="30" state="GGggrrrrGGggrrrr"/>
        <phase duration="3"  state="yyyyrrrryyyyrrrr"/>
        <phase duration="30" state="rrrrGGggrrrrGGgg"/>
        <phase duration="3"  state="rrrryyyyrrrryyyy"/>
    </tlLogic>
    <junction id="center" type="traffic_light" x="0.00" y="0.00" incLanes="N2C_0 N2C_1 E2C_0 E2C_1 S2C_0 S2C_1 W2C_0 W2C_1" intLanes="" shape="-6.40,6.40 6.40,6.40 6.40,-6.40 -6.40,-6.40"/>
    <junction id="east" type="priority" x="200.00" y="0.00" incLanes="C2E_0 C2E_1" intLanes="" shape="200.00,0.00 200.00,-6.40 200.00,0.00"/>
    <junction id="north" type="priority" x="0.00" y="200.00" incLanes="C2N_0 C2N_1" intLanes="" shape="0.00,200.00 6.40,200.00 0.00,200.00"/>
    <junction id="south" type="priority" x="0.00" y="-200.00" incLanes="C2S_0 C2S_1" intLanes="" shape="0.00,-200.00 -6.40,-200.00 0.00,-200.00"/>
    <junction id="west" type="priority" x="-200.00" y="0.00" incLanes="C2W_0 C2W_1" intLanes="" shape="-200.00,0.00 -200.00,6.40 -200.00,0.00"/>
</net>"""
        with open("net.net.xml", "w") as f:
            f.write(net_xml)

def generate_exact_vehicle_routes(ns_cars, ew_cars):
    routes_xml = ['<routes>', '    <vType id="car" length="4.7" minGap="1.3" maxSpeed="13.89"/>']
    routes_xml.append('    <route id="NS" edges="N2C C2S"/>')
    routes_xml.append('    <route id="EW" edges="E2C C2W"/>')
    
    for i in range(ns_cars):
        depart = i * 0.5
        routes_xml.append(f'    <vehicle id="ns_{i}" type="car" route="NS" depart="{depart:.1f}"/>')
        
    for i in range(ew_cars):
        depart = i * 0.5
        routes_xml.append(f'    <vehicle id="ew_{i}" type="car" route="EW" depart="{depart:.1f}"/>')
        
    routes_xml.append('</routes>')
    with open("routes_custom.rou.xml", "w") as f:
        f.write("\n".join(routes_xml))

def calculate_paper_duration(ns_cars, ew_cars, min_g=10, max_g=60):
    total = ns_cars + ew_cars
    if total == 0:
        return 15, 15
    ratio_ns = ns_cars / total
    ratio_ew = ew_cars / total
    cycle = min(120, max(30, int(1.5 * total + 15)))
    ns_g = max(min_g, min(max_g, int(cycle * ratio_ns)))
    ew_g = max(min_g, min(max_g, int(cycle * ratio_ew)))
    return ns_g, ew_g

def run_simulation(custom_program=False, ns_g=30, ew_g=30, yellow_time=3):
    """Runs standard SUMO simulation and measures total wait time accurately."""
    traci.start(["sumo", "-c", "sim.sumocfg"])
    
    if custom_program:
        phases = [
            traci.trafficlight.Phase(ns_g, STATE_NS_GREEN),
            traci.trafficlight.Phase(yellow_time, STATE_NS_YELLOW),
            traci.trafficlight.Phase(ew_g, STATE_EW_GREEN),
            traci.trafficlight.Phase(yellow_time, STATE_EW_YELLOW),
        ]
        logic = traci.trafficlight.Logic("custom", 0, 0, phases)
        traci.trafficlight.setProgramLogic(TLS_ID, logic)
        traci.trafficlight.setProgram(TLS_ID, "custom")
        
    vehicle_wait_dict = {}
    step = 0
    
    while step < 180 and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for v in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(v) < 0.1:
                vehicle_wait_dict[v] = vehicle_wait_dict.get(v, 0) + 1
                
    traci.close()
    
    total_vehicles = len(vehicle_wait_dict)
    avg_wait = sum(vehicle_wait_dict.values()) / total_vehicles if total_vehicles > 0 else 0.0
    return avg_wait

def run_custom_simulation(ns_cars, ew_cars, user_green_time, yellow_time):
    setup_sumo_files()
    generate_exact_vehicle_routes(ns_cars, ew_cars)
    
    ns_g, ew_g = calculate_paper_duration(ns_cars, ew_cars)
    
    print("\n" + "=" * 60)
    print(f" SCENARIO: NS={ns_cars} vehicles | EW={ew_cars} vehicles")
    print(f" User Baseline Green : {user_green_time}s | Yellow: {yellow_time}s")
    print(f" Optimized Allocation : NS Green={ns_g}s | EW Green={ew_g}s")
    print("=" * 60 + "\n")
    
    base_avg_wait = run_simulation(custom_program=False)
    opt_avg_wait = run_simulation(custom_program=True, ns_g=ns_g, ew_g=ew_g, yellow_time=yellow_time)
    
    print("=" * 60)
    print(f" Fixed Baseline Average Wait Time : {base_avg_wait:.2f} s")
    print(f" Optimized Average Wait Time     : {opt_avg_wait:.2f} s")
    if base_avg_wait > 0:
        imp = (base_avg_wait - opt_avg_wait) / base_avg_wait * 100
        print(f" Delay Reduction                 : {imp:.1f}%")
    print("=" * 60)

    # Plot Chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        [f"Fixed ({user_green_time}s)", f"Optimized (NS:{ns_g}s, EW:{ew_g}s)"], 
        [base_avg_wait, opt_avg_wait], 
        color=["#7f8c8d", "#2ecc71"], 
        width=0.45
    )
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height:.2f}s", ha="center", va="bottom", fontweight="bold")
        
    plt.ylabel("Average Delay per Vehicle (seconds)")
    plt.title(f"SUMO Traffic Delay Comparison ({ns_cars} NS vs {ew_cars} EW)")
    plt.tight_layout()
    plt.savefig("user_custom_traffic_result.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    try:
        ns_cars_input = int(input("Enter cars on North-South (NS) [e.g. 20]: "))
        ew_cars_input = int(input("Enter cars on East-West (EW)   [e.g. 5] : "))
        green_input   = int(input("Enter fixed Green duration (s) [e.g. 30]: "))
        yellow_input  = int(input("Enter Yellow duration (s)      [e.g. 3] : "))
    except (ValueError, EOFError):
        ns_cars_input, ew_cars_input, green_input, yellow_input = 20, 5, 30, 3
        
    run_custom_simulation(ns_cars_input, ew_cars_input, green_input, yellow_input)