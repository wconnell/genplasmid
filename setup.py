from setuptools import setup, find_packages

setup(
    name="genplasmid",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython",
        # Add other dependencies here
    ],
    author="William Connell",
    author_email="wconnell93@gmail.com",
    description="Investigating generative models for plasmid design",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wconnell/genplasmid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)