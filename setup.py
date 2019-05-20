from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SeisNN',
    packages=['seisnn'],
    version='0.0.1dev1',
    description='Deep learning seismic phase picking package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    author='jimmy',
    author_email='jimmy60504@gmail.com',
    url='https://github.com/jimmy60504/SeisNN',
    python_requires='>=3.5',
    install_requires=[
        'matplotlib>=3.0.3',
        'numpy>=1.16.2',
        'scipy>=1.2.1',
        'obspy>=1.1.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 2 - Pre-Alpha"
    ]
)
