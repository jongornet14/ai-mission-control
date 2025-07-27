from setuptools import setup, find_packages

setup(
    name="intellinaut",
    version="1.0.0",
    author="Jonathan Gornet",
    author_email="jonathan.gornet@gmail.com",
    description="Reinforcement Learning Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jongornet14/ai-mission-control",
    packages=find_packages(),
    install_requires=[
        # Your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'intellinaut=intellinaut.cli.commands:main',
            'intellinaut-train=intellinaut.cli.commands:train_command',
            'intellinaut-worker=intellinaut.cli.commands:worker_command',
            'intellinaut-coordinator=intellinaut.cli.commands:coordinator_command',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        'intellinaut': ['configs/*.yaml', 'configs/**/*.yaml'],
    },
)