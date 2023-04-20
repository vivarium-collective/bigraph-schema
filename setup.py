import re
from setuptools import setup, find_packages


VERSION = '0.0.5'


with open("README.md", "r") as readme:
    description = readme.read()


setup(
    name="bigraph-schema",
    version=VERSION,
    author="Eran Agmon, Ryan Spangler",
    author_email="agmon.eran@gmail.com, ryan.spangler@gmail.com",
    description="A serializable type schema for compositional systems biology",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivarium-collective/bigraph-schema",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
    install_requires=[
        # List your package dependencies here
        "parsimonious",
        "fire",
    ],
)
