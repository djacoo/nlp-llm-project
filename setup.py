"""
Setup script for NLP LLM Project
"""

from setuptools import setup, find_packages

setup(
    name="nlp-llm-project",
    version="1.0.0",
    description="Email routing system using Large Language Models",
    author="NLP Project Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "scikit-learn>=1.0.0",
        "evaluate>=0.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "wandb>=0.15.0",
        ],
    },
)
