from setuptools import setup
from psyleague.psyleague import __version__

setup(
    name = 'psyleague',
    packages = ['psyleague'],
    package_data = {'psyleague': ['psyleague.cfg']},
    version = __version__,
    license = 'MIT',
    description = "Local tester for Topcoder Marathons & AtCoder Heuristic Contests",
    author = 'Psyho',
    url = 'https://github.com/FakePsyho/psyleague',
    keywords = ['CompetitiveProgramming', 'CodinGame', 'Tester'],
    install_requires=['tabulate', 'trueskill', 'openskill', 'portalocker', 'toml'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    entry_points = {'console_scripts': ['psyleague = psyleague.psyleague:_main']},
)