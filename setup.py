from distutils.core import setup

from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'pyinfer',
    packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "examples", "docs", "out", "dist","media","test"]),
    version = '0.0.3',
    license='Apache-2.0',
    description = 'Pyinfer is a model agnostic Python utility tool for ML developers and researchers to benchmark model inference statistics.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Charles Pierse',
    author_email = 'charlespierse@gmail.com',
    url = 'https://github.com/cdpierse/pyinfer',
    keywords = ['machine learning', 'inference', 'benchmark', 'model inference', 'model serving'],
    install_requires=[
        'tabulate>=0.8.7',
        'psutil>=5.7.3'
    ],
    extras_require={
        "Plotting": ["matplotlib>=3.3.2"],
        "all": ["matplotlib>=3.3.2"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.8',
    ],
)
