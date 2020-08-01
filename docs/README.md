# Building docs

Use [Sphinx](https://www.sphinx-doc.org/en/master/) to build your docs automatically.

## Install sphinx

    pip install Sphinx 

## Run sphinx-quickstart

[Sphinx Quickstart](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-quickstart.html)

## Pick a theme

[Sphinx Themes](https://sphinx-themes.org/)

Install theme

    pip install sphinx-bootstrap-theme

Change in [conf.py](conf.py)

    html_theme = 'bootstrap' 

## Autosummary templates 

Check out the [link](https://stackoverflow.com/a/62613202) to generate autodoc recursively. 

## Deploy to Github Pages

[Deploying documentation to GitHub Pages with continuous integration](https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/)

Check out [.circleci/config.yml](/.circleci/config.yml) for example