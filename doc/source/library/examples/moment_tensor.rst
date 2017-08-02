Moment Tensor Operations
========================

Transformations to and from Moment Tensor using the :py:class:`pyrocko.moment_tensor` modules.
Conversion between Global Centroid Moment tensors format with pointing upward, southward, and eastward (Up-South-East; USE) system to geographic Cartesian tensor (North-Eeast-Down; NED) transforms by:

.. math::
    :nowrap:

    \begin{align*} 
        M_{rr} &= M_{dd}, & M_{  r\theta} &= M_{nd},\\
        M_{\theta\theta} &= M_{ nn}, & M_{r\phi} &= -M_{ed},\\
        M_{\phi\phi} &=  M_{ee}, & M_{\theta\phi} &= -M_{ne}
    \end{align*}


Convert Moment Tensor components to strike, dip and rake
--------------------------------------------------------

Moment Tensor construction is shown by using the moment components
and conversion to strike, dip and rake.


Download :download:`moment_tensor_example1.py </../../src/tutorials/moment_tensor_example1.py>`

.. literalinclude :: /../../src/tutorials/moment_tensor_example1.py
    :language: python


Strike, dip and rake to Moment Tensor
-------------------------------------

Conversion from strike, dip and rake to the Moment Tensor. Afterwards
we normalize the Moment Tensor. 

Download :download:`moment_tensor_example2.py </../../src/tutorials/moment_tensor_example2.py>`

.. literalinclude :: /../../src/tutorials/moment_tensor_example2.py
    :language: python
