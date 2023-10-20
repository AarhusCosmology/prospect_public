from setuptools import setup

setup(
    name='prospect',
    version='23.0.0',
    packages=['prospect'],
    install_requires=['numpy>=1.17.0', 'scipy>=1.5', 'PyYAML>=5.1', 'GetDist>=1.3.1', 'mpi4py'],
    entry_points={
        'console_scripts': [
            'prospect=prospect.run:run_from_shell',
            'prospect-analyse=prospect.run:analyse_from_shell',
            'prospect-reanneal=prospect.profile:reanneal_from_shell'
        ],
    },
)


