{{ name | escape | underline }}

.. In this template, methods are ignored even if the parent class
   has only a private version of the method. For example, if the parent
   class has a method "_cross_entropy", and the child class has a method
   "cross_entropy", that method will not be documented by this template.
   This fixes an issue with inheritance, for example from
   tensorflow_probability.substrates.jax.distributions

   The modules liesel.tfp.jax.distributions and
   liesel.tfp.numpy.distributions are both affected

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% block methods %}
   {% if methods | reject("in", inherited_members) | reject("equalto", "__init__") | list | length > 0 %}
   .. rubric:: {{ _('Methods') }}

   This section contains does not show inherited methods. Please refer to the
   documentation of the parent class for documentation on inherited methods.

   .. autosummary::
      :toctree:

   {% for item in methods %}
   {% if not (item == "__init__" or item in inherited_members) %}
   {% if not "_" + item in inherited_members %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes | reject("in", inherited_members) | list | length > 0 %}
   .. rubric:: {{ _('Attributes') }}

   This section contains does not show inherited attributes. Please refer to the
   documentation of the parent class for documentation on inherited attributes.

   .. autosummary::
      :template: autosummary/attribute.rst
      :toctree:

      {% for item in attributes %}
      {% if not item in inherited_members %}
         ~{{ name }}.{{ item }}
      {% endif %}
      {%- endfor %}

   {% endif %}

   {% endblock %}
