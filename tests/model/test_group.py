import pytest

from liesel.model import Data, Group, Var


class TestGroup:
    def test_init(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test", var1=v1, var2=v2)

        assert "var1" in g
        assert "var2" in g
        assert g.name in v1.groups
        assert g.name in v2.groups

    def test_getitem(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test", var1=v1, var2=v2)
        assert g["var1"] == v1
        assert g["var2"] == v2

    def test_vars(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g = Group("test", var1=v1, var2=v2, nd1=n1)

        assert len(g.vars) == 2
        assert "var1" in g.vars
        assert "var2" in g.vars

    def test_nodes(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g = Group("test", var1=v1, var2=v2, nd1=n1)

        assert len(g.nodes) == 1
        assert "nd1" in g.nodes

    def test_duplicate_name(self) -> None:
        """A node cannot be member of two groups with the same name."""
        v1 = Var(0.0, name="v1")
        _ = Group("g1", var1=v1)

        with pytest.raises(RuntimeError):
            Group("g1", var1=v1)
