# Plan

1. Allow model to re-build graph
    a) Method for custom re-building
    b) LieselInterface calls this method

2. Update previously forbidden methods
    a) Node.add_inputs
    b) Node.set_inputs

3. Update previously forbidden properties
    a) Node.name (setter)
    b) Node.needs_seed (setter)
    c) Calc.function
    d) Dist.at
    e) Dist.distribution
    f) Dist.per_obs
    g) Var.dist_node
        * Also errors if replacement dist_node has a model
    h) Var.name
    i) Var.observed
    j) Var.parameter
    k) Var.value_node

4. Model.locked (if yes, raise old error, default true; error message includes warning about changing behavior in v0.5).

5. LieselInterface calls Model.build_graph()

6. Replace node with var

7. Replace var with node or float

8. Replace with named and unnamed nodes or vars
