{{ fullname | escape | underline }}

name {{ name }}
fullname {{ fullname }}
objname {{ objname }}
module {{ module }}
classes {{ classes }}
class {{ class }}

{% block classes %}
{% if classes %}

{% for class in classes %}


.. autoclass :: {{ fullname }}.{{ class }}
    :show-inheritance:
    :members:
    :undoc-members:

{%- endfor %}
{% endif %}
{% endblock %}
