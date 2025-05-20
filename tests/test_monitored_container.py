import pytest
import simpy
from app import MonitoredContainer

def test_monitored_container_initial_state():
    """Test the initial state of MonitoredContainer"""
    env = simpy.Environment()
    container = MonitoredContainer(env, init=100)
    
    print(container.levels[0][1])
    
    assert container.get_quarterly_level(0) == 100
    assert len(container.inflows) == 0
    assert len(container.outflows) == 0
    assert len(container.levels) == 1
    assert container.levels[0] == (0, 100)

def test_monitored_container_quarterly_flows():
    """Test quarterly flow tracking"""
    env = simpy.Environment()
    container = MonitoredContainer(env, init=100)

    def test_process():
        # Quarter 1: Add 50, remove 20
        yield env.timeout(1)
        yield container.put(50)
        yield container.get(20)
        
        # Quarter 2: Add 30, remove 40
        yield env.timeout(1)
        yield container.put(30)
        yield container.get(40)
        
        # Quarter 3: Add 60, remove 10
        yield env.timeout(1)
        yield container.put(60)
        yield container.get(10)
        
    env.process(test_process())
    env.run()
    
    print(container.inflows)
    print(container.outflows)
    print(container.levels)
    
    # Test Quarter 1 (time 0)
    q1_inflows, q1_outflows = container.get_quarterly_flows(1)
    assert q1_inflows == 50
    assert q1_outflows == 20
    assert container.get_quarterly_level(1) == 130
    
    # Test Quarter 2 (time 1)
    q2_inflows, q2_outflows = container.get_quarterly_flows(2)
    assert q2_inflows == 30
    assert q2_outflows == 40
    assert container.get_quarterly_level(2) == 120
    
    # Test Quarter 3 (time 2)
    q3_inflows, q3_outflows = container.get_quarterly_flows(3)
    assert q3_inflows == 60
    assert q3_outflows == 10
    assert container.get_quarterly_level(3) == 170

def test_monitored_container_total_flows():
    """Test total flow tracking"""
    env = simpy.Environment()
    container = MonitoredContainer(env, init=100)
    
    def test_process():
        yield env.timeout(1)
        yield container.put(50)
        yield container.get(20)
        yield env.timeout(1)
        yield container.put(30)
        yield container.get(40)
    
    env.process(test_process())
    env.run()
    
    print(container.inflows)
    print(container.outflows)
    print(container.levels)
    
    total_inflows = sum(container.inflows)
    total_outflows = sum(container.outflows)
    assert total_inflows == 80
    assert total_outflows == 60
    assert container.get_quarterly_level(2) == 120  # 100 + 80 - 60