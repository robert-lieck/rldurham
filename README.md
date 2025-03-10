# Reinforcement Learning Durham

[![tests](https://github.com/robert-lieck/rldurham/actions/workflows/tests.yml/badge.svg)](https://github.com/robert-lieck/rldurham/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/robert-lieck/rldurham/branch/main/graph/badge.svg?token=XAUCWNS7II)](https://codecov.io/gh/robert-lieck/rldurham)

![build](https://github.com/robert-lieck/rldurham/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/rldurham.svg)](https://badge.fury.io/py/rldurham)

[![doc](https://github.com/robert-lieck/rldurham/actions/workflows/doc.yml/badge.svg)](https://robert-lieck.github.io/rldurham/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for the Reinforcement Learning courses at Durham University.

See the [notebooks](https://github.com/robert-lieck/rldurham/tree/main/notebooks) and [gallery](https://robert-lieck.github.io/rldurham/auto_examples/index.html) for some examples.

## Installation

Install/upgrade via terminal as

```bash
pip install swig
pip install --upgrade rldurham
```

or Jupyter notebook as

```python
!pip install swig
!pip install --upgrade rldurham
```

## Known Issues

- `swig` has to be installed **before** `rldurham` (as above) because of dependency issues with the packages used by `rldurham` in particular `gymnasium`.
- Under python `3.11`, `swig` may need to be installed using system tools (i.e. not via `pip`).