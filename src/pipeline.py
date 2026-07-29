"""The stock-and-flow housing pipeline model."""

import simpy

from .containers import MonitoredContainer

class HousingPipeline:
    def __init__(self, env, init_planning, init_approved, init_started, application_rate, approval_rate, start_rate, completion_rate, 
                 planning_success_rate, approved_to_start_rate, start_to_completion_rate, init_completed=0):
        self.env = env
        # Containers for each stage and initial amounts
        self.in_planning = MonitoredContainer(env, init=init_planning)
        self.approved_not_started = MonitoredContainer(env, init=init_approved)
        self.started = MonitoredContainer(env, init=init_started)
        self.completed = MonitoredContainer(env, init=init_completed)
        
        # Flow rates (application rate -- absolute value, or: expected length of time to progress in quarters)
        self.application_rate = application_rate  # Keep this fixed as input
        self.approval_rate = approval_rate
        self.start_rate = start_rate
        self.completion_rate = completion_rate
        
        # Success rates (percentage that successfully moves to next stage)
        self.planning_success_rate = planning_success_rate
        self.approved_to_start_rate = approved_to_start_rate
        self.start_to_completion_rate = start_to_completion_rate
        
        # Start processes
        self.env.process(self.application_flow())
        self.env.process(self.approval_flow())
        self.env.process(self.start_flow())
        self.env.process(self.completion_flow())
    
    def application_flow(self):
        while True:
            yield self.in_planning.put(self.application_rate)
            yield self.env.timeout(1)
    
    def approval_flow(self):
        while True:
            amount = self.in_planning.level * 1/self.approval_rate
            successful_amount = amount * self.planning_success_rate
            yield self.in_planning.get(amount)
            yield self.approved_not_started.put(successful_amount)
            yield self.env.timeout(1)
    
    def start_flow(self):
        while True:
            amount = self.approved_not_started.level * 1/self.start_rate
            successful_amount = amount * self.approved_to_start_rate
            yield self.approved_not_started.get(amount)
            yield self.started.put(successful_amount)
            yield self.env.timeout(1)
    
    def completion_flow(self):
        while True:
            amount = self.started.level * 1/self.completion_rate
            successful_amount = amount * self.start_to_completion_rate
            yield self.started.get(amount)
            yield self.completed.put(successful_amount)
            yield self.env.timeout(1)

