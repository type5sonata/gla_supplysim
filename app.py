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

def generate_quarter_labels(start_year, start_quarter, num_quarters):
    labels = []
    current_year = start_year
    current_quarter = start_quarter
    
    for _ in range(num_quarters):
        labels.append(f"{current_year} Q{current_quarter}")
        current_quarter += 1
        if current_quarter > 4:
            current_quarter = 1
            current_year += 1
    
    return labels

def run_simulation(simulation_length, init_planning, init_approved, init_started, init_completed,
                  application_rate, approval_rate, start_rate, completion_rate,
                  planning_success_rate, approved_to_start_rate, start_to_completion_rate,
                  start_year, start_quarter):
    
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
    quarters = generate_quarter_labels(start_year, start_quarter, simulation_length)
    applications = []
    approvals = []
    starts = []
    completions = []
    
    # Previous values for calculating flows
    prev_planning = init_planning
    prev_approved = init_approved
    prev_started = init_started
    prev_completed = init_completed
    
    # Run simulation and collect data
    for quarter in range(simulation_length):
        # Run simulation for one quarter
        env.run(until=quarter + 1)
        
        # Calculate flows (quarterly changes)
        applications.append(application_rate)  # This is already a flow
        
        # Calculate successful approvals
        approval_amount = prev_planning * approval_rate
        successful_approvals = approval_amount * planning_success_rate
        approvals.append(successful_approvals)
        
        # Calculate successful starts
        start_amount = prev_approved * start_rate
        successful_starts = start_amount * approved_to_start_rate
        starts.append(successful_starts)
        
        # Calculate successful completions
        completion_amount = prev_started * completion_rate
        successful_completions = completion_amount * start_to_completion_rate
        completions.append(successful_completions)
        
        # Update previous values
        prev_planning = pipeline.in_planning.level
        prev_approved = pipeline.approved_not_started.level
        prev_started = pipeline.started.level
        prev_completed = pipeline.completed.level
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Quarter': quarters,
        'Applications': applications,
        'Approvals': approvals,
        'Starts': starts,
        'Completions': completions
    })
    
    return results

# Streamlit UI
st.title('Housing Pipeline Simulation Dashboard')

# Sidebar for parameters
st.sidebar.header('Simulation Parameters')

# Simulation timing
st.sidebar.subheader('Simulation Timing')
start_year = st.sidebar.number_input('Start Year', value=2025, min_value=2024, max_value=2050)
start_quarter = st.sidebar.selectbox('Start Quarter', [1, 2, 3, 4], index=0)
simulation_length = st.sidebar.number_input('Simulation Length (quarters)', value=12, min_value=1, max_value=100)

# Flow times
st.sidebar.subheader('Average Time in Each Stage (quarters)')
application_rate = st.sidebar.number_input('New Applications per Quarter', value=1000, step=100)
planning_time = st.sidebar.number_input('Average Time to Get Planning Permission', value=15.0, min_value=1.0, step=0.1, help='Average number of quarters it takes to get planning permission')
start_time = st.sidebar.number_input('Average Time from Approval to Start', value=6.8, min_value=1.0, step=0.1, help='Average number of quarters between approval and construction start')
completion_time = st.sidebar.number_input('Average Time from Start to Completion', value=11.8, min_value=1.0, step=0.1, help='Average number of quarters from construction start to completion')

# Success rates
st.sidebar.subheader('Success Rates')
planning_success_rate = st.sidebar.slider('Planning Success Rate', 0.0, 1.0, 0.8, help='Proportion of planning applications that are successful')
approved_to_start_rate = st.sidebar.slider('Approved to Start Rate', 0.0, 1.0, 0.9, help='Proportion of approved developments that successfully start construction')
start_to_completion_rate = st.sidebar.slider('Start to Completion Rate', 0.0, 1.0, 0.95, help='Proportion of started constructions that successfully complete')

# Convert times to rates before running simulation
if st.sidebar.button('Run Simulation'):
    # Convert average times to rates (rate = 1/time)
    approval_rate = 1.0 / planning_time
    start_rate = 1.0 / start_time
    completion_rate = 1.0 / completion_time
    
    # Run simulation with calculated rates
    results = run_simulation(
        simulation_length=simulation_length,
        init_planning=50000,
        init_approved=115000,
        init_started=188000,
        init_completed=0,
        application_rate=application_rate,
        approval_rate=approval_rate,
        start_rate=start_rate,
        completion_rate=completion_rate,
        planning_success_rate=planning_success_rate,
        approved_to_start_rate=approved_to_start_rate,
        start_to_completion_rate=start_to_completion_rate,
        start_year=start_year,
        start_quarter=start_quarter
    )
    
    # Add average time information to the dashboard
    st.sidebar.subheader('Implied Quarterly Rates')
    st.sidebar.text(f'Approval Rate: {approval_rate:.3f}')
    st.sidebar.text(f'Start Rate: {start_rate:.3f}')
    st.sidebar.text(f'Completion Rate: {completion_rate:.3f}')
    
    # Create plots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Quarterly Flow Rates', 'Annual Totals', 'Average Quarterly Flows'),
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.3, 0.3])
    
    # Flow rates over time (Quarterly)
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Applications'],
                  name='Applications', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Approvals'],
                  name='Approvals', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Starts'],
                  name='Starts', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Quarter'], y=results['Completions'],
                  name='Completions', line=dict(color='red')),
        row=1, col=1
    )
    
    # Calculate annual totals
    results['Year'] = [q.split()[0] for q in results['Quarter']]
    annual_totals = results.groupby('Year').sum()
    
    # Annual totals bar chart
    fig.add_trace(
        go.Bar(x=annual_totals.index, 
               y=annual_totals['Applications'],
               name='Applications (Annual)',
               marker_color='blue',
               showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=annual_totals.index, 
               y=annual_totals['Approvals'],
               name='Approvals (Annual)',
               marker_color='green',
               showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=annual_totals.index, 
               y=annual_totals['Starts'],
               name='Starts (Annual)',
               marker_color='orange',
               showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=annual_totals.index, 
               y=annual_totals['Completions'],
               name='Completions (Annual)',
               marker_color='red',
               showlegend=False),
        row=2, col=1
    )
    
    # Average quarterly flows (last subplot)
    quarters_to_show = min(4, len(results))  # Show last 4 quarters or all if less
    last_quarters = results.iloc[-quarters_to_show:]
    
    fig.add_trace(
        go.Bar(x=['Applications', 'Approvals', 'Starts', 'Completions'],
               y=[last_quarters['Applications'].mean(),
                  last_quarters['Approvals'].mean(),
                  last_quarters['Starts'].mean(),
                  last_quarters['Completions'].mean()],
               name='Average Flow Rate (Last 4 Quarters)',
               marker_color=['blue', 'green', 'orange', 'red']),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(height=1200, showlegend=True, barmode='group')
    
    # Update axes labels
    fig.update_xaxes(title_text='Quarter', row=1, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_xaxes(title_text='Stage', row=3, col=1)
    
    fig.update_yaxes(title_text='Number of Units per Quarter', row=1, col=1)
    fig.update_yaxes(title_text='Number of Units per Year', row=2, col=1)
    fig.update_yaxes(title_text='Average Units per Quarter', row=3, col=1)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data with both quarterly and annual views
    st.subheader('Simulation Results')
    
    # Show quarterly results
    st.subheader('Quarterly Data')
    st.dataframe(results)
    
    # Show annual totals
    st.subheader('Annual Totals')
    st.dataframe(annual_totals)