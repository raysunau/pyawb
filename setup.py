from setuptools import find_packages, setup

requirements = [
    'pandas==1.1.1',
    'PyYAML==5.3.1',
    'scikit-learn==0.23.2'
]

setup_requirements = ['pytest-runner']
test_requirements = ['pytest>=3']

setup(
    python_requires='>=3.6',
    name='datk',
    packages=find_packages(include=['datk', 'datk.*']),
    version='0.1.0',
    description='Python Data Analytics library',
    author='RaySun',
    license='MIT',
    install_requires=requirements,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
)