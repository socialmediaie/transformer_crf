#!/usr/bin/env python
import os
from pathlib import Path

from setuptools import find_packages, setup

VERSION = (Path(__file__).parent / "VERSION").read_text()


install_requires = [
    "transformers",
    "datasets",
    "evaluate",
    "seqeval"
]

packages = find_packages(include=["transformer_crf*"], exclude=["tests", "dependencies"])
setup(
    name="transformer_crf",
    version=VERSION,
    install_requires=install_requires,
    setup_requires=["setuptools"],
    description="CRF Model for Transformer",
    packages=packages,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "pip-tools", 
            "pytest", 
            "pytest-asyncio"
        ], "notebook": [
            "jupyterlab",
        ],
    },
    python_requires=">=3.7",
    zip_safe=False,
    package_data={
        "transformer_crf": [
        ]
    },
)