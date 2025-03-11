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

Start with a clean Python 3.10 environment (recommended) and install/upgrade via terminal as

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

- **Have you tried turning it off and on again?** Restarting your kernel and/or restarting the install from a fresh and clean Python 3.10 environment resolves most problems.
- **Check the error messages!** In many cases, they provide useful information and in many cases the problem is not with `rldurham` but either a general Python problem or a problem with `gymnasium`.
- On **NCC** you need to create your own custom environment/kernel (as for the deep learning coursework); you cannot `pip install ...` things in the default environment (it may first look as if you can, but then the packages cannot be found).
- Problems related to `swig`
  - Remember that `swig` has to be **explicitly installed before** `rldurham` (as above) because of dependency issues with the packages used by `rldurham`, in particular `gymnasium`.
  - Under python `3.11`, `swig` may need to be installed using system tools (i.e. not via `pip`), better avoid `3.11` and use `3.10` instead.
  - If you see errors mentioning `Box2D`, this is related to `gymnasium`, which requires `Box2D`, which requires `swig`, which frequently causes problems. Installing from a clean environment, *first* installing `swig`, and using Python 3.10 (avoiding higher versions) are the best ways to avoid these problems.