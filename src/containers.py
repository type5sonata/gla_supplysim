"""SimPy container subclass that records its own flow history."""

import simpy

class MonitoredContainer(simpy.Container):
    def __init__(self, env, init=0, capacity=float('inf')):
        self.env = env
        super().__init__(env, capacity=capacity, init=init)
        # Track inflows and outflows separately with timestamps
        self.inflows = []  # List of (time, amount) tuples for inflows
        self.outflows = []  # List of (time, amount) tuples for outflows
        self.levels = [(env.now, init)]  # List of (time, level) tuples
        
    def get(self, amount):
        # Record the outflow before the change
        self.outflows.append(amount)
        # Perform the get operation
        result = super().get(amount)
        # Record the new level after the change
        self.levels.append((self.env.now, self.level))
        return result
        
    def put(self, amount):
        # Record the inflow before the change
        self.inflows.append(amount)
        # Perform the put operation
        result = super().put(amount)
        # Record the new level after the change
        self.levels.append((self.env.now, self.level))
        return result
    
    def get_quarterly_flows(self, quarter):
        """Get total inflows and outflows for a specific quarter"""
        # Include flows that occur exactly at start_time but exclude those at end_time
        return self.inflows[quarter-1], self.outflows[quarter-1]
    
    def get_quarterly_level(self, quarter):
        if quarter == 0:
            return self.levels[0][1]
        else:
            """Get the level at a specific quarter"""
            return self.levels[quarter*2][1]
