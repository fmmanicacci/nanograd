from setuptools import find_packages, setup

setup(
    name='nanograd',
    packages=find_packages(include=['nanograd']),
    version='0.1.0',
    description="""A tiny scalar-valued autograd engine inspired from the one created by
    Karpathy""",
    author='François-Marie Manicacci',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.2.1'],
    test_suite='tests',
)
