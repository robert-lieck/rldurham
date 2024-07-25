# PythonTemplatePackage

[![tests](https://github.com/robert-lieck/pythontemplatepackage/actions/workflows/tests.yml/badge.svg)](https://github.com/robert-lieck/pythontemplatepackage/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/robert-lieck/pythontemplatepackage/branch/main/graph/badge.svg?token=XAUCWNS7II)](https://codecov.io/gh/robert-lieck/pythontemplatepackage)

![build](https://github.com/robert-lieck/pythontemplatepackage/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/pythontemplatepackage.svg)](https://badge.fury.io/py/pythontemplatepackage)

[![doc](https://github.com/robert-lieck/pythontemplatepackage/actions/workflows/doc.yml/badge.svg)](https://robert-lieck.github.io/pythontemplatepackage/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A template repo for Python packages featuring:
- `main`/`dev` branch workflow
- unittests and code coverage
- publishing the package on PyPi
- building documentation and publishing via GitHub pages


## How To

To create a new Python package from this template, start by cloning this repo (or use it as a template when creating a new repo on GitHub) and then follow the procedure outlined below.

***There now is the [`rename.py`](https://github.com/robert-lieck/pythontemplatepackage/blob/main/pythontemplatepackage/rename.py) script for initialising/renaming everything for a new package!***

Go to the `pythontemplatepackage` folder and run

```shell
python rename.py -d /path/to/your/repo
```

to interactively make any required changes. You need to add the `-a` switch to actually apply any changes, otherwise a dry run is performed and changes are printed to the terminal for you to review. Use `-h` to print a help menu.

You should remove the `.rename_test_dir`, which is only for testing the `rename.py` script.

### Badges README

The `README.md` is obviously specific to your project, but you might want to use the badges at the top.
- The `tests`, `build`, and `doc` badge show the success status of the respective GitHub actions. The easiest is to follow the procedure below and update them afterwards.
- The `codecov` badge should be replaced by the one specific to your package (see [Tests](#Tests) below).
- In the `pypi` badge the package name needs to be adapted. After the first successful upload (see [PyPi](#PyPi) below) it will show the correct version and link to the PyPi page.
- In the `doc` badge, you may want to link to the actual documentation (as is done above) instead of the GitHub action (as is the default).

### Package Name

The example package provided by this repo is named `pythontemplatepackage` and this name appears in many locations. Therefore, the first step is to choose a package name (check that it is available on PyPi if you plan to publish it there!) and replace all occurrences by the name of your package. In particular, you have to rename the folder `pythontemplatepackage` accordingly and replace all occurrences in the following files (this is described in more detail in the respective sections below):
- `setup.py`
- `tests/test_template.py`
- `.github/workflows/tests.yml`
- `.github/workflows/test_dev.yml`
- `doc/conf.py`
- `doc/index.rst`
- `doc/api_summary.rst`

### Folder Structure

- Your source code goes into the `pythontemplatepackage` directory (after renaming it to your package name).
- Your unittests go into the `test` directory.
- Your documentation goes into the `doc` directory.
- The `.github/workflows` folder contains `*.yml` files that define GitHub actions that
  - run tests on the `main` and `dev` branch (see [Tests](#Tests))
  - publish the package on [pypi.org](https://pypi.org/) (see [PyPi](#PyPi))
  - build the documentation and publish it via GitHub pages (see [Documentation](#Documentation))

### Adapt `requirements.txt` and `setup.py`

List all required Python packages in `requirements.txt`.

In `setup.py` replace the following:
- `name="pythontemplatepackage"`: replace with the name of your package
- `version="..."`: the version of your package
- `author="..."`: your name
- `author_email="..."`: your email
- `description="..."`: a short description of the package
- `url="..."`: the URL of the repo
- `python_requires="..."`: the Python version requirement for your package

Moreover, in the `classifiers` argument, you may want to adapt the following to your liking:
- `Programming Language :: Python :: 3`
- `License :: OSI Approved :: GNU General Public License v3 (GPLv3)`
- `Operating System :: OS Independent`

If you change the license information, you probably also want to adapt the `LICENSE` file and the badge at the top of the `README.md`.

### Tests

Replace the `test_template.py` file with some real tests for you package (at least, you have to replace `pythontemplatepackage` with your package name for things to work).

In `tests.yml` (for `main` branch) and `test_dev.yml` (for `dev` branch) adapt the following:
- `os: [ubuntu-latest, macos-latest, windows-latest]`: operating systems to test for
- `python-version: ["3.9", "3.10"]`: Python versions to test for
- `pythontemplatepackage`: the name of your package chosen above
- `Upload coverage to Codecov`: you can delete this section if you do not want to use [codecov.io](https://about.codecov.io/) (remember to also remove the codecov badge above)
  - If you use codecov, you will have to:
    - enable the project in your account
    - add the `CODECOV_TOKEN` to your repository's action secrets to be able to upload reports
    - get the correct coverage badge after the first report has been uploaded under `Settings > Badges & Graphs` (the link includes a token).

The GitHub actions for running tests on the `main` and `dev` branch are almost identical. The only differences are:
- their name (used to display in the web interface)
- the branch name (adapt if you use different names)
- tests on `main` also upload code coverage reports
- the test and codecov badge refer the tests on `main`

The tests run on `push` and `pull_request` events of the respective branch or when triggered manually.

### PyPi

You have to set up an API token to be able to upload to PyPi:
- In you [PyPi account page](https://pypi.org/manage/account/) create a new API token valid for all projects (will be changed later).
- In the repository's GitHub page under `Settings > Secrets > Actions` create a new _Repository Secret_ with name `PYPI_API_TOKEN` and copy-paste the PyPi token (`pypi-...`).
- _After_ the  first successful upload, _change_ that token by one that is specific to this package (for security reasons).

### Documentation

The `doc` folder contains a skeleton documentation using the [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/) that you can adapt to your needs. You should replace the following:
- in `conf.py`, `index.rst`, `api_summary.rst`
  - replace `pythontemplatepackage` with your package name
- in `conf.py` adapt the following:
  - `project = 'pythontemplatepackage'`
  - `copyright = '...'`
  - `author = '...'`

#### Local Builds

For local builds, you can run `make` commands in the `doc` directory (you will have to install the packages specified in `doc/requirements.txt`), in particular
- `make html`: builds the documentation
- `make doctest`: runs all code examples in the documentation and checks if the actual output matches the one shown in the documentation
- `make clean`: remove all built files (including `_autosummary` and `auto_examples`)
- `make help`: get information about available make commands.

#### Publish via GitHub Pages

To publish the documentation via GitHub pages, you have to:
- create the `gh-pages` branch
- enable GitHub pages on `gh-pages` branch using the `/` (root) directory.

The `doc` action builds the documentation via `make html` and pushes it to the `gh-pages` branch. It also runs `make linkcheck` and `make doctest` to check for missing links and test examples in the documentation.