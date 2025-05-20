import pytest
import simpy
from app import MonitoredContainer

@pytest.fixture
def env():
    return simpy.Environment()

@pytest.fixture
def container(env):
    return MonitoredContainer(env, init=100)
