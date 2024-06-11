from setuptools import setup
__version__ = '0.4.1'

setup(
    name = 'psyleague',
    packages = ['psyleague'],
    package_data = {'psyleague': ['psyleague.cfg']},
    version = __version__,
    license = 'MIT',
    description = "Local league system for bot contests.",
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