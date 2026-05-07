from pathlib import Path

from setuptools import find_packages, setup

requirements = (Path(__file__).parent / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="xfpad",
    version="1.0.0",
    description="X-FPAD: Fingerprint PAD Exposimeter — reference implementation.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Simone Carta, Roberto Casula, Gian Luca Marcialis",
    author_email="roberto.casula@unica.it",
    url="https://github.com/carsim97/xfpad",
    license="MIT",
    packages=find_packages(exclude=("scripts", "configs", "data", "outputs", "checkpoints")),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
