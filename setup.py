from setuptools import setup, find_packages

setup(
    name="unsupervised-learning-images",
    version="1.0.0",
    description="Projekt autoencodera z klasteryzacją i inpaintingiem obrazów",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "scikit-learn>=1.1.0",
        "umap-learn>=0.5.3",
        "comet_ml>=3.31.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)