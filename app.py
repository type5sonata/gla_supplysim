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

def run_pipeline_simulation(name, simulation_length, init_planning, init_approved, init_started, init_completed,
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

def create_pipeline_parameters(label, col):
    col.subheader(f'{label} Parameters')
    
    # Flow times
    application_rate = col.number_input(f'New Applications per Quarter ({label})', 
                                      value=1000 if label == "Large Private Sites" else 500, 
                                      step=100, key=f'app_rate_{label}')
    
    planning_time = col.number_input(f'Average Time to Get Planning Permission ({label})', 
                                   value=15.0 if label == "Large Private Sites" else 12.0, 
                                   min_value=1.0, step=0.1, 
                                   help='Average number of quarters it takes to get planning permission',
                                   key=f'plan_time_{label}')
    
    start_time = col.number_input(f'Average Time from Approval to Start ({label})', 
                                value=6.8 if label == "Large Private Sites" else 5.0, 
                                min_value=1.0, step=0.1,
                                help='Average number of quarters between approval and construction start',
                                key=f'start_time_{label}')
    
    completion_time = col.number_input(f'Average Time from Start to Completion ({label})', 
                                     value=11.8 if label == "Large Private Sites" else 8.0, 
                                     min_value=1.0, step=0.1,
                                     help='Average number of quarters from construction start to completion',
                                     key=f'comp_time_{label}')
    
    # Success rates
    planning_success_rate = col.slider(f'Planning Success Rate ({label})', 
                                     0.0, 1.0, 0.8,
                                     help='Proportion of planning applications that are successful',
                                     key=f'plan_success_{label}')
    
    approved_to_start_rate = col.slider(f'Approved to Start Rate ({label})', 
                                      0.0, 1.0, 0.9,
                                      help='Proportion of approved developments that successfully start construction',
                                      key=f'app_start_{label}')
    
    start_to_completion_rate = col.slider(f'Start to Completion Rate ({label})', 
                                        0.0, 1.0, 0.95,
                                        help='Proportion of started constructions that successfully complete',
                                        key=f'start_comp_{label}')
    
    # Initial values
    col.subheader(f'{label} Initial Values')
    init_planning = col.number_input(f'Initial Planning ({label})', 
                                   value=50000 if label == "Large Private Sites" else 25000,
                                   step=1000,
                                   key=f'init_plan_{label}')
    
    init_approved = col.number_input(f'Initial Approved ({label})', 
                                   value=115000 if label == "Large Private Sites" else 50000,
                                   step=1000,
                                   key=f'init_app_{label}')
    
    init_started = col.number_input(f'Initial Started ({label})', 
                                  value=188000 if label == "Large Private Sites" else 75000,
                                  step=1000,
                                  key=f'init_start_{label}')
    
    return {
        'application_rate': application_rate,
        'planning_time': planning_time,
        'start_time': start_time,
        'completion_time': completion_time,
        'planning_success_rate': planning_success_rate,
        'approved_to_start_rate': approved_to_start_rate,
        'start_to_completion_rate': start_to_completion_rate,
        'init_planning': init_planning,
        'init_approved': init_approved,
        'init_started': init_started
    }

# Streamlit UI
st.title('Housing Pipeline Simulation Dashboard')

# Sidebar for parameters
st.sidebar.header('Simulation Parameters')

# Simulation timing
st.sidebar.subheader('Simulation Timing')
start_year = st.sidebar.number_input('Start Year', value=2025, min_value=2024, max_value=2050)
start_quarter = st.sidebar.selectbox('Start Quarter', [1, 2, 3, 4], index=0)
simulation_length = st.sidebar.number_input('Simulation Length (quarters)', value=12, min_value=1, max_value=100)

# Create three columns for pipeline parameters
col1, col2, col3 = st.columns(3)

# Get parameters for each pipeline
large_private_params = create_pipeline_parameters("Large Private Sites", col1)
small_private_params = create_pipeline_parameters("Small Private Sites", col2)
public_params = create_pipeline_parameters("Public Sites", col3)

if st.button('Run Simulation'):
    # Run simulations for each pipeline type
    pipeline_results = {}
    for name, params in [
        ("Large Private Sites", large_private_params),
        ("Small Private Sites", small_private_params),
        ("Public Sites", public_params)
    ]:
        # Convert times to rates
        approval_rate = 1.0 / params['planning_time']
        start_rate = 1.0 / params['start_time']
        completion_rate = 1.0 / params['completion_time']
        
        # Run simulation
        results = run_pipeline_simulation(
            name=name,
            simulation_length=simulation_length,
            init_planning=params['init_planning'],
            init_approved=params['init_approved'],
            init_started=params['init_started'],
            init_completed=0,
            application_rate=params['application_rate'],
            approval_rate=approval_rate,
            start_rate=start_rate,
            completion_rate=completion_rate,
            planning_success_rate=params['planning_success_rate'],
            approved_to_start_rate=params['approved_to_start_rate'],
            start_to_completion_rate=params['start_to_completion_rate'],
            start_year=start_year,
            start_quarter=start_quarter
        )
        pipeline_results[name] = results
    
    # Create combined results
    combined_results = pipeline_results["Large Private Sites"].copy()
    for col in ['Applications', 'Approvals', 'Starts', 'Completions']:
        combined_results[col] = (
            pipeline_results["Large Private Sites"][col] +
            pipeline_results["Small Private Sites"][col] +
            pipeline_results["Public Sites"][col]
        )
    
    # Create tabs for different views
    tab_all, tab_large, tab_small, tab_public = st.tabs([
        "All Sites Combined",
        "Large Private Sites",
        "Small Private Sites",
        "Public Sites"
    ])
    
    def create_plots(results, title_prefix=""):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{title_prefix}Quarterly Flow Rates',
                f'{title_prefix}Annual Totals',
                f'{title_prefix}Average Quarterly Flows'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Flow rates over time (Quarterly)
        for name, color in [
            ('Applications', 'blue'),
            ('Approvals', 'green'),
            ('Starts', 'orange'),
            ('Completions', 'red')
        ]:
            fig.add_trace(
                go.Scatter(
                    x=results['Quarter'],
                    y=results[name],
                    name=name,
                    line=dict(color=color)
                ),
                row=1, col=1
            )
        
        # Calculate annual totals
        results['Year'] = [q.split()[0] for q in results['Quarter']]
        annual_totals = results.groupby('Year').sum()
        
        # Annual totals bar chart
        for name, color in [
            ('Applications', 'blue'),
            ('Approvals', 'green'),
            ('Starts', 'orange'),
            ('Completions', 'red')
        ]:
            fig.add_trace(
                go.Bar(
                    x=annual_totals.index,
                    y=annual_totals[name],
                    name=f'{name} (Annual)',
                    marker_color=color,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Average quarterly flows
        quarters_to_show = min(4, len(results))
        last_quarters = results.iloc[-quarters_to_show:]
        
        fig.add_trace(
            go.Bar(
                x=['Applications', 'Approvals', 'Starts', 'Completions'],
                y=[
                    last_quarters['Applications'].mean(),
                    last_quarters['Approvals'].mean(),
                    last_quarters['Starts'].mean(),
                    last_quarters['Completions'].mean()
                ],
                name='Average Flow Rate (Last 4 Quarters)',
                marker_color=['blue', 'green', 'orange', 'red']
            ),
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
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45, row=1, col=1)
        
        return fig, annual_totals
    
    # Display plots and data in tabs
    with tab_all:
        st.subheader('Combined Results (All Sites)')
        fig, annual_totals = create_plots(combined_results)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Quarterly Data')
        st.dataframe(combined_results)
        st.subheader('Annual Totals')
        st.dataframe(annual_totals)
    
    with tab_large:
        st.subheader('Large Private Sites Results')
        fig, annual_totals = create_plots(
            pipeline_results["Large Private Sites"],
            "Large Private Sites - "
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Quarterly Data')
        st.dataframe(pipeline_results["Large Private Sites"])
        st.subheader('Annual Totals')
        st.dataframe(annual_totals)
    
    with tab_small:
        st.subheader('Small Private Sites Results')
        fig, annual_totals = create_plots(
            pipeline_results["Small Private Sites"],
            "Small Private Sites - "
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Quarterly Data')
        st.dataframe(pipeline_results["Small Private Sites"])
        st.subheader('Annual Totals')
        st.dataframe(annual_totals)
    
    with tab_public:
        st.subheader('Public Sites Results')
        fig, annual_totals = create_plots(
            pipeline_results["Public Sites"],
            "Public Sites - "
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Quarterly Data')
        st.dataframe(pipeline_results["Public Sites"])
        st.subheader('Annual Totals')
        st.dataframe(annual_totals)