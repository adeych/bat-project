from setuptools import setup, find_packages

setup(
    name="pipistrelle_bat_analysis",
    version="0.1.0",
    # Automatically finds 'models' and 'testing' because they have __init__.py
    packages=find_packages(),
    
    # Core dependencies
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "soundfile",
        "tqdm",
        "torch",
        "torchaudio",
        "iterative-stratification", # for iterative stratification
    ],
    
    # Optional: Dependencies for specific encoders
    extras_require={
        "perch": ["tensorflow", "tensorflow-hub"],
        "beats": ["librosa"],
    },

    author="Anastasia Deych",
    description="Audio Dataset and Preprocessing Pipeline for Pipistrelle Bat Recordings",
    python_requires=">=3.11",
)