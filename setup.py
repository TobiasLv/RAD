from setuptools import setup, find_packages

setup(
    name='pytorch-rad',
    version='0.1.3',
    author='iDLab',
    author_email='lyo.tobias@foxmail.com',
    description='A pytorch implementation of the optimizer RAD',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TobiasLv/RAD',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
    ],
)
