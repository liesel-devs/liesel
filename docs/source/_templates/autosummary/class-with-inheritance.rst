{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% block methods %}
   {% if methods | reject("equalto", "__init__") | list %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:

   {% for item in methods %}
   {% if not item == "__init__" %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: autosummary/attribute.rst
      :toctree:

      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {%- endfor %}

   {% endif %}

   {% endblock %}
