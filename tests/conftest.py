import pytest
import simpy
from src.containers import MonitoredContainer

@pytest.fixture
def env():
    return simpy.Environment()

@pytest.fixture
def container(env):
    return MonitoredContainer(env, init=100)
