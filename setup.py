from setuptools import setup, find_packages

setup(
    name='set2set',
    version='0.1.0',
    packages=find_packages(where='set2set'),
    url = 'https://github.com/arunppsg/Set2Set',
    author = 'Arun Thiagarajan',
    license='MIT',
    description = 'Utility for representing sets in an order invariant manner',
    install_requires=[
        'torch'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
  ],
    python_requires='>=3.7'
)
