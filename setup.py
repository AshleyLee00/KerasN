# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "KerasN - Neural Network Implementation in Python"

setup(
    name="kerasN",
    version="0.3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.2",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "pytest>=6.0.0",
        "tqdm>=4.50.0",
        "pandas",  # pandas도 유지
    ],
    author="Yeseol Lee",
    author_email="your.email@example.com",
    description="A Neural Network Implementation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kerasN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 