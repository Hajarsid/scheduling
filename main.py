import argparse
import logging
import sys


from numpy import random
from des import SchedulerDES
from schedulers import FCFS, SJF, RR, SRTF
from DQN_temp import BrainDQN

# default values
seed = int.from_bytes(random.bytes(4), byteorder="little")
num_processes =25
arrivals_per_time_unit = 200.0
avg_cpu_burst_time = 2
quantum = 0.5
context_switch_time = 0.0
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# parse arguments
parser = argparse.ArgumentParser(description='NOSE2 AE2: Discrete Event Simulation')
parser.add_argument('--seed', '-S', help='PRNG random seed value', type=int)
parser.add_argument('--processes', '-P', help='Number of processes to simulate', default=num_processes, type=int)
parser.add_argument('--arrivals', '-L', help='Avg number of process arrivals per time unit',
                    default=arrivals_per_time_unit, type=float)
parser.add_argument('--cpu_time', '-c', help='Avg duration of CPU burst', default=avg_cpu_burst_time, type=float)
parser.add_argument('--cs_time', '-x', help='Duration of each context switch', default=context_switch_time, type=float)
parser.add_argument('--quantum', '-q', help='Duration of each quantum (Round Robin scheduling)', default=quantum,
                    type=float)
parser.add_argument('--verbose', '-v', help='Turn logging on; specify multiple times for more verbosity',
                    action='count')
args = parser.parse_args()
if args.seed:
    seed = args.seed
if args.verbose == 1:
    logging.getLogger().setLevel(logging.INFO)
elif args.verbose is not None:
    logging.getLogger().setLevel(logging.DEBUG)
num_processes = args.processes
arrivals_per_time_unit = args.arrivals
avg_cpu_burst_time = args.cpu_time
context_switch_time = args.cs_time
quantum = args.quantum

print("NOSE2 :: AE2 :: Scheduler Discrete Event Simulation")
print("---------------------------------------------------")

# print input specification
print("Using seed: " + str(seed))
base_sim = SchedulerDES(num_processes=num_processes, arrivals_per_time_unit=arrivals_per_time_unit,
                        avg_cpu_burst_time=avg_cpu_burst_time)
base_sim.generate_and_init(seed)
print("Processes to be executed:")
base_sim.print_processes()

# instantiate simulators
simulators = [FCFS(num_processes=num_processes, arrivals_per_time_unit=arrivals_per_time_unit,
                   avg_cpu_burst_time=avg_cpu_burst_time, context_switch_time=context_switch_time),
              SJF(num_processes=num_processes, arrivals_per_time_unit=arrivals_per_time_unit,
                  avg_cpu_burst_time=avg_cpu_burst_time, context_switch_time=context_switch_time),
              RR(num_processes=num_processes, arrivals_per_time_unit=arrivals_per_time_unit,
                 avg_cpu_burst_time=avg_cpu_burst_time, context_switch_time=context_switch_time, quantum=quantum),
              SRTF(num_processes=num_processes, arrivals_per_time_unit=arrivals_per_time_unit,
                   avg_cpu_burst_time=avg_cpu_burst_time, context_switch_time=context_switch_time)]

# run simulators
for sim in simulators:
    print("-----")
    print(sim.full_name() + ":")
    logging.info("--- " + sim.full_name() + " ---")
    sim.run(seed)
    sim.print_statistics()

lamda = [round(0.1*i/2.0,2) for i in range(1,9)]
actions = len(lamda)

Loss = []
Success = []
Fre = []

noise = 3
num_sensor = 10  # N
policy = 2  # choose power change policy for PU, it should be 1(Multi-step) or 2(Single step)

brain = BrainDQN(actions, num_sensor)
com = SchedulerDES(num_processes, arrivals_per_time_unit, avg_cpu_burst_time, context_switch_time=0.0,
                   quantum=math.inf)
terminal = True
recording = 100000

while (recording > 0):
    # initialization
    if (terminal == True):
        com.ini()
        observation0, reward0, terminal = com.frame_step(np.zeros(actions), policy, False)
        brain.setInitState(observation0)

    # train
    action, recording = brain.getAction()
    nextObservation, reward, terminal = com.frame_step(action, policy, True)
    loss = brain.setPerception(nextObservation, action, reward)

    # test
    if (recording + 1) % 500 == 0:

        Loss.append(loss)
        print("iteration : %d , loss : %f ." % ((100000 - recording, loss)))

        success = 0.0
        fre = 0
        num = 1000.0
        for ind in range(1000):
            T = 0
            com.ini_test()
            observation0_test, reward_test, terminal_test = com.frame_step_test(np.zeros(actions), policy, False)
            while (terminal_test != True) and T < 20:
                action_test = brain.getAction_test(observation0_test)
                observation0_test, reward_test, terminal_test = com.frame_step_test(action_test, policy, True)
                T += 1
            if terminal_test == True:
                success += 1
                fre += T
        if success == 0:
            fre = 0
        else:
            fre = fre / success
        success = success / num
        Success.append(success)
        Fre.append(fre)
        print("success : %f , step : %f ." % ((success, fre)))

plt.plot(Loss)
plt.show()

plt.plot(Success)
plt.show()

plt.plot(Fre)
plt.show()