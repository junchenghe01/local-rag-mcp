"""Minimal setup.py for editable install compatibility."""
from setuptools import setup, find_packages

setup(
    name="rag-mcp",
    version="2.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
