from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SeisNN',
    packages=['seisnn'],
    version='0.0.1dev1',
    description='Deep learning seismic picking package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='jimmy',
    author_email='jimmy60504@gmail.com',
    url='https://github.com/jimmy60504/SeisNN',
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
