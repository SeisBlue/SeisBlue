"""
SeisNN - Deep learning seismic phase picking project
"""
import argparse
import datetime
import setuptools

version = '0.5.0'

parser = argparse.ArgumentParser()
parser.add_argument('--dev', help='Dev build', action='store_true')
args = parser.parse_args()
if args.dev:
    version += 'dev' + datetime.datetime.now().strftime("%Y%m%d")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='SeisNN',
    packages=['seisnn'],
    version=version,
    description='Deep learning seismic phase picking project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    author='jimmy',
    author_email='jimmy60504@gmail.com',
    url='https://github.com/SeisNN/SeisNN',
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 2 - Pre-Alpha"
    ]
)
