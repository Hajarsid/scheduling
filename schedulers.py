from des import SchedulerDES
import numbers
from enum import Enum, auto

from event import Event, EventTypes
from process import Process, ProcessStates


class FCFS(SchedulerDES):
    # is given the first (earliest) event from the events queue. It should inspect that event and/or the list of
    # processes (self.processes) , select one of those in the READY state that should be the next to execute,
    # and return it. The function should of course select the next process based on whatever scheduler algorithm you
    # implement (FIFO/SJF/SRTF/RR).
    def scheduler_func(self, cur_event):
        for cur_process in self.processes:
            if cur_event.process_id == cur_process.process_id:
                return cur_process

    def dispatcher_func(self, cur_process):
        cur_process.process_state = ProcessStates.RUNNING
        st = cur_process.service_time
        cur_process.run_for(st, self.time)
        cur_process.process_state = ProcessStates.TERMINATED
        new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_DONE,
                          event_time=cur_process.departure_time)
        return new_event


class SJF(SchedulerDES):
    def scheduler_func(self, cur_event):
        ready_process = [i for i in self.processes if i.process_state == ProcessStates.READY]
        firstobject = True
        for i, sorting in enumerate(self.processes):
            if sorting.process_state == ProcessStates.READY:
                if firstobject == True:
                    firstobject = False
                    minim = sorting.service_time
                    i_min = i
                elif minim > sorting.service_time:
                    minim = sorting.service_time
                    i_min = i

        cur_process = self.processes[i_min]
        return cur_process

    def dispatcher_func(self, cur_process):
        cur_process.process_state = ProcessStates.RUNNING
        st = cur_process.service_time
        cur_process.run_for(st, self.time)
        cur_process.process_state = ProcessStates.TERMINATED
        new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_DONE,
                          event_time=cur_process.departure_time)
        return new_event


class RR(SchedulerDES):
    def scheduler_func(self, cur_event):
        for cur_process in self.processes:
            if cur_event.process_id == cur_process.process_id:
                return cur_process

    def dispatcher_func(self, cur_process):
        cur_process.process_state = ProcessStates.RUNNING
        st = cur_process.remaining_time

        if st <= self.quantum:
            cur_process.run_for(st, self.time)
            cur_process.process_state = ProcessStates.TERMINATED
            new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_DONE,
                              event_time=cur_process.departure_time)
        else:
            cur_process.run_for(self.quantum, self.time)
            cur_process.process_state = ProcessStates.READY
            new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_REQ,
                              event_time=self.time + self.quantum)

        return new_event


class SRTF(SchedulerDES):
    def scheduler_func(self, cur_event):
        ready_process = [i for i in self.processes if i.process_state == ProcessStates.READY]
        firstobject = True
        for i, sorting in enumerate(self.processes):
            if sorting.process_state == ProcessStates.READY:
                if firstobject == True:
                    firstobject = False
                    minim = sorting.remaining_time
                    i_min = i
                elif minim > sorting.remaining_time:
                    minim = sorting.remaining_time
                    i_min = i

        cur_process = self.processes[i_min]
        return cur_process
    def dispatcher_func(self, cur_process):
        cur_process.process_state = ProcessStates.RUNNING
        st = cur_process.remaining_time
        next_event=self.next_event_time()
        timenow=self.time
        if  next_event>= st+timenow:
            cur_process.run_for(st, timenow)
            cur_process.process_state = ProcessStates.TERMINATED
            new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_DONE,
                              event_time=cur_process.departure_time)
        else:
            cur_process.run_for(next_event-timenow, timenow)
            cur_process.process_state = ProcessStates.READY
            new_event = Event(process_id=cur_process.process_id, event_type=EventTypes.PROC_CPU_REQ,
                              event_time=next_event)

        return new_event