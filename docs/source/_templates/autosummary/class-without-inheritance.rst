{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% block methods %}
   {% if methods | reject("in", inherited_members) | reject("equalto", "__init__") | list |length > 0 %}
   .. rubric:: {{ _('Methods') }}

   This section is empty if this class has only inherited attributes.

   .. autosummary::
      :toctree:

   {% for item in methods %}
   {% if not (item == "__init__" or item in inherited_members) %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes | reject("in", inherited_members) | list | length > 0 %}
   .. rubric:: {{ _('Attributes') }}

   This section is empty if this class has only inherited attributes.

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
