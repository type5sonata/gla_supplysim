import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Default thread configurations
DEFAULT_THREADS = {
    "Large Private Sites": {
        "application_rate": 6455,
        "planning_time": 9.3,
        "start_time": 9.4,
        "completion_time": 28.3,
        "planning_success_rate": 0.9,
        "approved_to_start_rate": 0.7,
        "start_to_completion_rate": 0.9,
        "init_planning": 59743,
        "init_approved": 53769,
        "init_started": 152570
    },
    "Small Private Sites": {
        "application_rate": 1760,
        "planning_time": 9.3,
        "start_time": 10.8,
        "completion_time": 24.9,
        "planning_success_rate": 0.9,
        "approved_to_start_rate": 0.7,
        "start_to_completion_rate": 0.9,
        "init_planning": 16427,
        "init_approved": 14785,
        "init_started": 41964
    },
    "Public Sites": {
        "application_rate": 1289,
        "planning_time": 9.3,
        "start_time": 15.5,
        "completion_time": 25.6,
        "planning_success_rate": 0.99,
        "approved_to_start_rate": 0.99,
        "start_to_completion_rate": 0.9,
        "init_planning": 11560,
        "init_approved": 11445,
        "init_started": 32475
    }
}

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

def apply_policies_to_parameters(base_params, policies, current_quarter_label, development_type):
    """Apply active policies to modify the base parameters"""
    modified_params = base_params.copy()
    
    # Get year and quarter from the current quarter label (e.g., "2025 Q1")
    current_year = int(current_quarter_label.split()[0])
    current_quarter = int(current_quarter_label.split()[1][1])
    
    # Apply each policy that is active for the current quarter
    for policy in policies:
        policy_year = int(policy['year'])
        policy_quarter = int(policy['quarter'])
        
        # Skip if policy doesn't apply to this development type
        if policy.get('target_type', 'All') != "All" and policy.get('target_type') != development_type:
            continue
        
        # Check if policy is active (current time is at or after policy start)
        if (current_year > policy_year) or (current_year == policy_year and current_quarter >= policy_quarter):
            param_name = policy['parameter']
            change_type = policy['change_type']
            change_value = float(policy['change_value'])
            
            # Apply the change directly since we've already handled the time conversion
            if change_type == 'Absolute':
                modified_params[param_name] += change_value
            elif change_type == 'Multiply':
                modified_params[param_name] *= change_value
    
    return modified_params

def run_pipeline_simulation(name, simulation_length, init_planning, init_approved, init_started, init_completed,
                          application_rate, approval_rate, start_rate, completion_rate,
                          planning_success_rate, approved_to_start_rate, start_to_completion_rate,
                          start_year, start_quarter, policies=[]):
    
    # Initialize environment
    env = simpy.Environment()
    
    # Create pipeline with custom initial values
    pipeline = HousingPipeline(env, init_planning, init_approved, init_started, init_completed, application_rate, approval_rate, start_rate, completion_rate, planning_success_rate, approved_to_start_rate, start_to_completion_rate)
    
    # Lists to store values at each timestep
    quarters = generate_quarter_labels(start_year, start_quarter, simulation_length)
    applications = []
    approvals = []
    starts = []
    completions = []
    
    # Lists to store stock values
    planning_stock = [init_planning]
    approved_stock = [init_approved]
    started_stock = [init_started]
    completed_stock = [init_completed]
    
    # Previous values for calculating flows
    prev_planning = init_planning
    prev_approved = init_approved
    prev_started = init_started
    prev_completed = init_completed
    
    # Base parameters
    base_params = {
        'application_rate': application_rate,
        'approval_rate': approval_rate,
        'start_rate': start_rate,
        'completion_rate': completion_rate,
        'planning_success_rate': planning_success_rate,
        'approved_to_start_rate': approved_to_start_rate,
        'start_to_completion_rate': start_to_completion_rate
    }
    
    # Run simulation and collect data
    for quarter_idx, quarter_label in enumerate(quarters):
        # Apply active policies for this quarter
        current_params = apply_policies_to_parameters(base_params, policies, quarter_label, name)
        
        # Set rates for this quarter
        pipeline.application_rate = current_params['application_rate']
        pipeline.approval_rate = current_params['approval_rate']
        pipeline.start_rate = current_params['start_rate']
        pipeline.completion_rate = current_params['completion_rate']
        pipeline.planning_success_rate = current_params['planning_success_rate']
        pipeline.approved_to_start_rate = current_params['approved_to_start_rate']
        pipeline.start_to_completion_rate = current_params['start_to_completion_rate']
        
        # Run simulation for one quarter
        env.run(until=quarter_idx + 1)
        
        # Append flows to lists.
        
        
        # Append end-of-quarter levels to lists.
        planning_stock.append(pipeline.in_planning.level)
        approved_stock.append(pipeline.approved_not_started.level)
        started_stock.append(pipeline.started.level)
        completed_stock.append(pipeline.completed.level)
    
    # Add this right before the DataFrame creation
    # print("\n=== Debug Information ===")
    # print(f"Number of quarters: {len(quarters)}")
    # print("\nList lengths:")
    # print(f"quarters: {len(quarters)}")
    # print(f"planning_stock: {len(planning_stock[:-1])}")
    # print(f"approved_stock: {len(approved_stock[:-1])}")
    # print(f"started_stock: {len(started_stock[:-1])}")
    # print(f"completed_stock: {len(completed_stock[:-1])}")
    # print("\nFlow records:")
    # print(f"in_planning.inflows: {len(pipeline.in_planning.inflows)}")
    # print(f"approved_not_started.inflows: {len(pipeline.approved_not_started.inflows)}")
    # print(f"started.inflows: {len(pipeline.started.inflows)}")
    # print(f"completed.inflows: {len(pipeline.completed.inflows)}")

    # Create DataFrame with results
    results = pd.DataFrame({
        'Quarter': quarters,
        'Applications': pipeline.in_planning.inflows, # Subsitute historical value for first quarter
        'Approvals': pipeline.approved_not_started.inflows,
        'Starts': pipeline.started.inflows,
        'Completions': pipeline.completed.inflows,
        'Planning Stock': planning_stock[:-1],
        'Approved Stock': approved_stock[:-1],
        'Started Stock': started_stock[:-1],
        'Completed Stock': completed_stock[:-1]
    })
    # Convert all numeric columns to integers
    numeric_columns = results.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    results[numeric_columns] = results[numeric_columns].astype(int)
    
    return results

def create_pipeline_parameters(label, col, defaults):
    col.subheader(f'{label} Parameters')

    # Flow times
    application_rate = col.number_input(f'New Applications per Quarter ({label})',
                                    value=defaults['application_rate'],
                                    step=100, key=f'app_rate_{label}')

    planning_time = col.number_input(f'Average Time to Get Planning Permission ({label})',
                                value=defaults['planning_time'],
                                min_value=1.0, step=0.1,
                                help='Average number of quarters it takes to get planning permission',
                                key=f'plan_time_{label}')

    start_time = col.number_input(f'Average Time from Approval to Start ({label})',
                                value=defaults['start_time'],
                                min_value=1.0, step=0.1,
                                help='Average number of quarters between approval and construction start',
                                key=f'start_time_{label}')

    completion_time = col.number_input(f'Average Time from Start to Completion ({label})',
                                    value=defaults['completion_time'],
                                    min_value=1.0, step=0.1,
                                    help='Average number of quarters from construction start to completion',
                                    key=f'comp_time_{label}')

    # Success rates
    planning_success_rate = col.slider(f'Planning Success Rate ({label})',
                                    0.0, 1.0, defaults['planning_success_rate'],
                                    help='Proportion of planning applications that are successful',
                                    key=f'plan_success_{label}')

    approved_to_start_rate = col.slider(f'Approved to Start Rate ({label})',
                                    0.0, 1.0, defaults['approved_to_start_rate'],
                                    help='Proportion of approved developments that successfully start construction',
                                    key=f'app_start_{label}')

    start_to_completion_rate = col.slider(f'Start to Completion Rate ({label})',
                                        0.0, 1.0, defaults['start_to_completion_rate'],
                                        help='Proportion of started constructions that successfully complete',
                                        key=f'start_comp_{label}')

    # Initial stock values
    col.subheader(f'{label} Initial Stock Parameters')
    init_planning = col.number_input(f'In planning ({label})',
                                value=defaults['init_planning'],
                                step=1000,
                                key=f'init_plan_{label}')

    init_approved = col.number_input(f'Approved not started ({label})',
                                value=defaults['init_approved'],
                                step=1000,
                                key=f'init_app_{label}')

    init_started = col.number_input(f'Started ({label})',
                                value=defaults['init_started'],
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

def main():
   
    # Streamlit UI
    st.title('Housing Pipeline Simulation Dashboard 1.0')

    # Create tabs for main interface and policy editor
    tab_main, tab_policies = st.tabs(["Main Dashboard", "Policy Editor"])

    with tab_policies:
        st.header("Policy Timeline Editor")
        st.markdown("""
        Add policies that will change parameters at specific points in time. 
        Each policy can modify a parameter in one of three ways:
        - Absolute: Change the parameter by an absolute value (positive or negative)
        - Multiply: Multiply the parameter by a value
        """)
        
        # Initialize session state for policies if it doesn't exist
        if 'policies' not in st.session_state:
            st.session_state.policies = []
        
        # Form for adding new policies
        with st.form("add_policy"):
            st.subheader("Add New Policy")
            
            # First row: Policy name and target type
            col_name, col_type = st.columns(2)
            policy_name = col_name.text_input("Policy Name", 
                                            value="New Policy",
                                            help="Give your policy a descriptive name")
            
            # Initialize threads if not present (for policy editor access before main tab)
            if 'threads' not in st.session_state:
                st.session_state.threads = list(DEFAULT_THREADS.keys())

            target_type = col_type.selectbox("Target Development Type",
                                           ["All"] + st.session_state.threads,
                                           help="Select which type of development this policy affects")
            
            # Second row: Timing, parameter, and change
            cols = st.columns(5)
            
            # Policy timing
            policy_year = cols[0].number_input("Year", min_value=2024, max_value=2050, value=2025)
            policy_quarter = cols[1].selectbox("Quarter", [1, 2, 3, 4], key="new_policy_quarter")
            
            # Parameter selection
            parameter = cols[2].selectbox("Parameter", [
                ("application_rate", "Number of applications"),
                ("approval_rate", "Planning Permission Time"),
                ("start_rate", "Approval to Start Time"),
                ("completion_rate", "Start to Completion Time"),
                ("planning_success_rate", "Planning Success Rate"),
                ("approved_to_start_rate", "Approved to Start Rate"),
                ("start_to_completion_rate", "Start to Completion Rate")
            ], format_func=lambda x: x[1])
            
            # Store the actual parameter name (first element of tuple)
            parameter_name = parameter[0]
            
            # Change type and value
            change_type = cols[3].selectbox("Change Type", ["Absolute", "Multiply"])
            change_value = cols[4].number_input("Change Value", value=0.0, step=0.1)

            # Submit button
            if st.form_submit_button("Add Policy"):
                new_policy = {
                    'name': policy_name,
                    'target_type': target_type,
                    'year': policy_year,
                    'quarter': policy_quarter,
                    'parameter': parameter_name,  # Use the actual parameter name
                    'change_type': change_type,
                    'change_value': change_value,
                    'display_value': change_value  # Store original input value for display
                }
                st.session_state.policies.append(new_policy)
                st.success("Policy added!")
        
        # Display and manage existing policies
        st.subheader("Existing Policies")
        if not st.session_state.policies:
            st.info("No policies added yet.")
        else:
            # Sort policies by year and quarter
            sorted_policies = sorted(
                enumerate(st.session_state.policies),
                key=lambda x: (x[1]['year'], x[1]['quarter'])
            )
            
            for i, (orig_idx, policy) in enumerate(sorted_policies):
                cols = st.columns([1, 4, 1])
                with cols[1]:
                    # Format the change value based on parameter type
                    if policy['parameter'] in ['approval_rate', 'start_rate', 'completion_rate']:
                        if policy['change_type'] == "Absolute":
                            value_display = f"{policy['display_value']} quarters"
                        else:
                            value_display = f"{policy['display_value']}{'%' if policy['change_type'] == 'Percentage' else 'x'}"
                    else:
                        value_display = f"{policy['display_value']}{'%' if policy['change_type'] == 'Percentage' else ''}"
                    
                    st.markdown(f"""
                    **{policy['name']}** (Policy {i+1})
                    - **When:** {policy['year']} Q{policy['quarter']}
                    - **Target:** {policy['target_type']}
                    - **Change:** {policy['change_type'].lower()} change to {policy['parameter']}: 
                      {value_display}
                    """)
                with cols[2]:
                    if st.button(f"Delete Policy {i+1}"):
                        st.session_state.policies.pop(orig_idx)
                        st.rerun()
        
        if st.button("Clear All Policies"):
            st.session_state.policies = []
            st.rerun()

    with tab_main:
        # Sidebar for parameters
        st.sidebar.header('Simulation Parameters')

        # Thread Manager section
        st.sidebar.subheader('Thread Manager')

        # Initialize session state for threads if not present
        if 'threads' not in st.session_state:
            st.session_state.threads = list(DEFAULT_THREADS.keys())

        # Multiselect for active threads
        active_threads = st.sidebar.multiselect(
            'Active Threads',
            options=list(DEFAULT_THREADS.keys()) + [t for t in st.session_state.threads if t not in DEFAULT_THREADS],
            default=st.session_state.threads,
            help='Select which threads to include in the simulation'
        )

        # Update session state with selected threads
        st.session_state.threads = active_threads

        # Add custom thread expander
        with st.sidebar.expander('Add Custom Thread'):
            new_thread_name = st.text_input('Thread Name', key='new_thread_name')
            if st.button('Add Thread'):
                if new_thread_name and new_thread_name not in st.session_state.threads:
                    st.session_state.threads.append(new_thread_name)
                    st.rerun()
                elif new_thread_name in st.session_state.threads:
                    st.warning('Thread already exists')
                else:
                    st.warning('Please enter a thread name')

        # Simulation timing
        st.sidebar.subheader('Simulation Timing')
        start_year = st.sidebar.number_input('Start Year', value=2024, min_value=2024, max_value=2050)
        start_quarter = st.sidebar.selectbox('Start Quarter', [1, 2, 3, 4], index=1)
        simulation_length = st.sidebar.number_input('Simulation Length (quarters)', value=23, min_value=1, max_value=100)

        # Dynamic columns for pipeline parameters
        thread_names = st.session_state.threads
        if thread_names:
            cols = st.columns(len(thread_names))
            thread_params = {}
            for i, name in enumerate(thread_names):
                # Use defaults from DEFAULT_THREADS if available, otherwise use Large Private Sites defaults
                defaults = DEFAULT_THREADS.get(name, DEFAULT_THREADS["Large Private Sites"])
                thread_params[name] = create_pipeline_parameters(name, cols[i], defaults)
        else:
            st.warning('No threads selected. Please add at least one thread in the sidebar.')
            thread_params = {}

        if st.button('Run Simulation'):
            if not thread_params:
                st.error('No threads to simulate. Please add at least one thread.')
            else:
                # Run simulations for each pipeline type
                pipeline_results = {}
                for name in st.session_state.threads:
                    params = thread_params[name]

                    # Run simulation with policies
                    results = run_pipeline_simulation(
                        name=name,
                        simulation_length=simulation_length,
                        init_planning=params['init_planning'],
                        init_approved=params['init_approved'],
                        init_started=params['init_started'],
                        init_completed=0,
                        application_rate=params['application_rate'],
                        approval_rate=params['planning_time'],
                        start_rate=params['start_time'],
                        completion_rate=params['completion_time'],
                        planning_success_rate=params['planning_success_rate'],
                        approved_to_start_rate=params['approved_to_start_rate'],
                        start_to_completion_rate=params['start_to_completion_rate'],
                        start_year=start_year,
                        start_quarter=start_quarter,
                        policies=[
                            {**p, 'target_type': p.get('target_type', 'All')}  # Ensure backward compatibility
                            for p in st.session_state.policies
                        ]  # Add policies to simulation
                    )
                    pipeline_results[name] = results

                # Create combined results
                first_thread = st.session_state.threads[0]
                combined_results = pipeline_results[first_thread].copy()
                for col in ['Applications', 'Approvals', 'Starts', 'Completions', 'Planning Stock', 'Approved Stock', 'Started Stock', 'Completed Stock']:
                    combined_results[col] = sum(
                        pipeline_results[name][col] for name in st.session_state.threads
                    )
            
                # Create tabs for different views dynamically
                tab_names = ["All Sites Combined"] + st.session_state.threads
                tabs = st.tabs(tab_names)

                def create_plots(results, title_prefix=""):
                    # Define GLA colors from the grid
                    GLA_COLORS = {
                        'dark_blue': '#4477AA',    # Top left
                        'light_blue': '#88CCEE',   # Top middle
                        'green': '#117733',        # Top right
                        'yellow': '#DDCC77',       # Bottom left
                        'red': '#CC6677',          # Bottom middle
                        'purple': '#AA4499'        # Bottom right
                    }

                    fig = make_subplots(
                        rows=3, cols=1,  # Changed from 4 to 3 rows
                        subplot_titles=(
                            f'{title_prefix}Quarterly Flow Rates',
                            f'{title_prefix}Annual Totals',
                            f'{title_prefix}Stock Evolution'
                        ),
                        vertical_spacing=0.1,
                        row_heights=[0.33, 0.33, 0.34]  # Adjusted row heights to fill space evenly
                    )

                    # Flow rates over time (Quarterly)
                    for name, color in [
                        ('Applications', GLA_COLORS['dark_blue']),
                        ('Approvals', GLA_COLORS['green']),
                        ('Starts', GLA_COLORS['yellow']),
                        ('Completions', GLA_COLORS['red'])
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
                    # Drop the Quarter column from annual totals since it's not meaningful
                    annual_totals = annual_totals.drop('Quarter', axis=1, errors='ignore')

                    # Annual totals line chart
                    for name, color in [
                        ('Applications', GLA_COLORS['dark_blue']),
                        ('Approvals', GLA_COLORS['green']),
                        ('Starts', GLA_COLORS['yellow']),
                        ('Completions', GLA_COLORS['red'])
                    ]:
                        fig.add_trace(
                            go.Scatter(
                                x=annual_totals.index,
                                y=annual_totals[name],
                                name=f'{name} (Annual)',
                                line=dict(color=color),
                                showlegend=False
                            ),
                            row=2, col=1
                        )

                    # Stock evolution
                    for name, color in [
                        ('Planning Stock', GLA_COLORS['dark_blue']),
                        ('Approved Stock', GLA_COLORS['green']),
                        ('Started Stock', GLA_COLORS['yellow']),
                        ('Completed Stock', GLA_COLORS['red'])
                    ]:
                        fig.add_trace(
                            go.Scatter(
                                x=results['Quarter'],
                                y=results[name],
                                name=f'{name}',
                                line=dict(color=color)
                            ),
                            row=3, col=1
                        )

                    # Update layout with light theme
                    fig.update_layout(
                        height=1500,
                        showlegend=True,
                        barmode='group',
                        plot_bgcolor='#f6f4f2',
                        paper_bgcolor='#f6f4f2',
                        font=dict(color='black')
                    )

                    # Update axes labels
                    fig.update_xaxes(title_text='Quarter', row=1, col=1, gridcolor='lightgrey')
                    fig.update_xaxes(title_text='Year', row=2, col=1, gridcolor='lightgrey')
                    fig.update_xaxes(title_text='Quarter', row=3, col=1, gridcolor='lightgrey')

                    fig.update_yaxes(title_text='Number of Units per Quarter', row=1, col=1, rangemode='nonnegative', gridcolor='lightgrey')
                    fig.update_yaxes(title_text='Number of Units per Year', row=2, col=1, rangemode='nonnegative', gridcolor='lightgrey')
                    fig.update_yaxes(title_text='Total Units in Stock', row=3, col=1, rangemode='nonnegative', gridcolor='lightgrey')

                    # Rotate x-axis labels
                    fig.update_xaxes(tickangle=45, row=1, col=1)
                    fig.update_xaxes(tickangle=45, row=3, col=1)

                    return fig, annual_totals

                # Display plots and data in tabs
                # First tab is combined results
                with tabs[0]:
                    st.subheader('Combined Results (All Sites)')
                    fig, annual_totals = create_plots(combined_results)
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader('Quarterly Data')
                    st.dataframe(combined_results)
                    st.subheader('Annual Totals')
                    st.dataframe(annual_totals)

                # Remaining tabs are individual threads
                for i, thread_name in enumerate(st.session_state.threads):
                    with tabs[i + 1]:
                        st.subheader(f'{thread_name} Results')
                        fig, annual_totals = create_plots(
                            pipeline_results[thread_name],
                            f"{thread_name} - "
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader('Quarterly Data')
                        st.dataframe(pipeline_results[thread_name])
                        st.subheader('Annual Totals')
                        st.dataframe(annual_totals)

if __name__ == "__main__":
    main()