"""Default calibration for each development thread.

These are the starting values shown in the sidebar. They were calibrated from
Molior and GLA planning data - see README for provenance."""



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
