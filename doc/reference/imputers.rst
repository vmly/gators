
.. _api.imputers:

********
Imputers
********

Two different types of imputers are available depending on the variable datatype,
namely: numerical and categorical (string or object).
 

.. note::

    * *NumericsImputer* imputes numerical variables.

    * *ObjectImputer* imputes only categorical variables.


Base Imputer
############
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   _BaseImputer

Imputers
########
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   NumericsImputer
   ObjectImputer

