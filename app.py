import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HousingPipeline:
    def __init__(self, env, init_planning=50000, init_approved=115000, init_started=188000, init_completed=0):
        self.env = env
        # Containers for each stage and initial amounts
        self.in_planning = simpy.Container(env, init=init_planning)
        self.approved_not_started = simpy.Container(env, init=init_approved)
        self.started = simpy.Container(env, init=init_started)
        self.completed = simpy.Container(env, init=init_completed)
        
        # Flow rates (proportion of containers processed per quarter)
        self.application_rate = 1000  # Keep this fixed as input
        self.approval_rate = 0.3      # Process 30% of planning applications per quarter
        self.start_rate = 0.25        # Start 25% of approved applications per quarter
        self.completion_rate = 0.2    # Complete 20% of started homes per quarter
        
        # Success rates (percentage that successfully moves to next stage)
        self.planning_success_rate = 0.8
        self.approved_to_start_rate = 0.9
        self.start_to_completion_rate = 0.95
        
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
            amount = self.in_planning.level * self.approval_rate
            successful_amount = amount * self.planning_success_rate
            yield self.in_planning.get(amount)
            yield self.approved_not_started.put(successful_amount)
            yield self.env.timeout(1)
    
    def start_flow(self):
        while True:
            amount = self.approved_not_started.level * self.start_rate
            successful_amount = amount * self.approved_to_start_rate
            yield self.approved_not_started.get(amount)
            yield self.started.put(successful_amount)
            yield self.env.timeout(1)
    
    def completion_flow(self):
        while True:
            amount = self.started.level * self.completion_rate
            successful_amount = amount * self.start_to_completion_rate
            yield self.started.get(amount)
            yield self.completed.put(successful_amount)
            yield self.env.timeout(1)

def run_simulation(simulation_length, init_planning, init_approved, init_started, init_completed,
                  application_rate, approval_rate, start_rate, completion_rate,
                  planning_success_rate, approved_to_start_rate, start_to_completion_rate):
    
    # Initialize environment
    env = simpy.Environment()
    
    # Create pipeline with custom initial values
    pipeline = HousingPipeline(env, init_planning, init_approved, init_started, init_completed)
    
    # Set custom rates
    pipeline.application_rate = application_rate
    pipeline.approval_rate = approval_rate
    pipeline.start_rate = start_rate
    pipeline.completion_rate = completion_rate
    pipeline.planning_success_rate = planning_success_rate
    pipeline.approved_to_start_rate = approved_to_start_rate
    pipeline.start_to_completion_rate = start_to_completion_rate
    
    # Lists to store values at each timestep
    quarters = []
    planning_levels = []
    approved_levels = []
    started_levels = []
    completed_levels = []
    
    # Run simulation and collect data
    for quarter in range(simulation_length):
        quarters.append(f"Q{quarter+1}")
        planning_levels.append(pipeline.in_planning.level)
        approved_levels.append(pipeline.approved_not_started.level)
        started_levels.append(pipeline.started.level)
        completed_levels.append(pipeline.completed.level)
        
        env.run(until=quarter + 1)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Quarter': quarters,
        'In Planning': planning_levels,
        'Approved Not Started': approved_levels,
        'Under Construction': started_levels,
        'Completed': completed_levels
    })
    
    return results

# Streamlit UI
st.title('Housing Pipeline Simulation Dashboard')

# Sidebar for parameters
st.sidebar.header('Simulation Parameters')

# Initial values
st.sidebar.subheader('Initial Values')
init_planning = st.sidebar.number_input('Initial Units in Planning', value=50000, step=1000)
init_approved = st.sidebar.number_input('Initial Units Approved Not Started', value=115000, step=1000)
init_started = st.sidebar.number_input('Initial Units Under Construction', value=188000, step=1000)
init_completed = st.sidebar.number_input('Initial Units Completed', value=0, step=1000)

# Flow rates
st.sidebar.subheader('Flow Rates (per quarter)')
application_rate = st.sidebar.number_input('New Applications Rate', value=1000, step=100)
approval_rate = st.sidebar.slider('Approval Processing Rate', 0.0, 1.0, 0.3)
start_rate = st.sidebar.slider('Start Rate', 0.0, 1.0, 0.25)
completion_rate = st.sidebar.slider('Completion Rate', 0.0, 1.0, 0.2)

# Success rates
st.sidebar.subheader('Success Rates')
planning_success_rate = st.sidebar.slider('Planning Success Rate', 0.0, 1.0, 0.8)
approved_to_start_rate = st.sidebar.slider('Approved to Start Rate', 0.0, 1.0, 0.9)
start_to_completion_rate = st.sidebar.slider('Start to Completion Rate', 0.0, 1.0, 0.95)

# Simulation length
simulation_length = st.sidebar.number_input('Simulation Length (quarters)', value=12, min_value=1, max_value=100)

# Run simulation button
if st.sidebar.button('Run Simulation'):
    # Run simulation with selected parameters
    results = run_simulation(
        simulation_length=simulation_length,
        init_planning=init_planning,
        init_approved=init_approved,
        init_started=init_started,
        init_completed=init_completed,
        application_rate=application_rate,
        approval_rate=approval_rate,
        start_rate=start_rate,
        completion_rate=completion_rate,
        planning_success_rate=planning_success_rate,
        approved_to_start_rate=approved_to_start_rate,
        start_to_completion_rate=start_to_completion_rate
    )
    
    # Create plots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Housing Pipeline Levels', 'Cumulative Completions'),
                        vertical_spacing=0.15)
    
    # Pipeline levels
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['In Planning'],
                  name='In Planning', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Approved Not Started'],
                  name='Approved Not Started', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Under Construction'],
                  name='Under Construction', line=dict(color='orange')),
        row=1, col=1
    )
    
    # Cumulative completions
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Completed'],
                  name='Completed', line=dict(color='red')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text='Quarter', row=2, col=1)
    fig.update_yaxes(title_text='Number of Units', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Completions', row=2, col=1)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data
    st.subheader('Simulation Results')
    st.dataframe(results) 