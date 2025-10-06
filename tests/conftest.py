# tests/conftest.py
import sys
import os

# This adds the main project directory to the Python path
# so that 'agent.py' can be imported by the test files.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))