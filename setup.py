from distutils.core import setup

from setuptools import find_packages

setup(
    name='pizza',
    version='1.0',
    py_modules=['pizza'],
    author='Dulan Jayasuriya',
    author_email='dulanjayasuriya@gmail.com',
    description='Pizza Classifier',
    url='https://github.com/CodeProcessor',
    license='MIT',
    packages=find_packages()
)
