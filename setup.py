from setuptools import find_packages, setup

setup(
    name="seahorse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    author="Tyler Romero",
    author_email="tyleraromero+seahorse@gmail.com",
    description="A small VLLM for research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tyler-romero/seahorse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
