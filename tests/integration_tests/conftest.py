import pytest


def pytest_addoption(parser):
    parser.addoption("--input_dir", action="store", default=None, help="Directory containing projects/ directory")
    parser.addoption("--credentials_path", action="store", default=None, help="Path to credentials JSON")
