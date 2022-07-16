{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
   {% if not fullname.startswith("liesel.tfp") %}
      :template: autosummary/class-without-inheritance.rst
   {% elif fullname.startswith("liesel.tfp.jax.distributions") %}
      :template: autosummary/class-without-inheritance-strict.rst
   {% elif fullname.startswith("liesel.tfp.numpy.distributions") %}
      :template: autosummary/class-without-inheritance-strict.rst
   {% else %}
      :template: autosummary/class-without-inheritance.rst
   {% endif %}
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   
   {% endif %}

   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
