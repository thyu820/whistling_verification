from setuptools import setup, find_packages

setup(
    name="whistling_verification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "librosa>=0.8.1",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="口哨聲聲者驗證系統",
)
