Moment tensor conversions
=========================

Transformations between different moment tensor representations using the
:py:class:`pyrocko.moment_tensor` module.


Convert moment tensor components to strike, dip and rake
--------------------------------------------------------

Moment tensor construction is shown by using the moment components (in
north-east-down coordinate system convention) and conversion to strike, dip and
rake. We also show how to extract the P-axis direction.

Download :download:`moment_tensor_example1.py </../../examples/moment_tensor_example1.py>`

.. literalinclude :: /../../examples/moment_tensor_example1.py
    :language: python


Strike, dip and rake to moment tensor
-------------------------------------

Conversion from strike, dip and rake to the moment tensor. Afterwards
we normalize the moment tensor. 

Download :download:`moment_tensor_example2.py </../../examples/moment_tensor_example2.py>`

.. literalinclude :: /../../examples/moment_tensor_example2.py
    :language: python
