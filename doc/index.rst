.. documentation master file; adapt to your liking (but should at least contain the root `toctree` directive).

Welcome to PythonTemplatePackage's documentation!
=================================================

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   auto_examples/index.rst
   api_summary

.. ...add more elements to table of contents

You can include code as part of the documentation

   >>> print("Hello World")
   Hello World

which can be tested by running ``make doctest``. This is also run by the GitHub action to build the documentation.

You can also include executable example files with code and text, which are shown in the :doc:`auto_examples/index`.
The ``test_examples.py`` unittest automatically runs these examples to check for errors.

.. autoclass:: pythontemplatepackage.myclass.MyClass
   :members:
   :noindex:


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
