"""Simulation driver: quarter labelling, policy application, and the run loop."""

import simpy
import pandas as pd

from .pipeline import HousingPipeline

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
