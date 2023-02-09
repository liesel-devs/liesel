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

    def test_setitem(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test")
        g["var1"] = v1
        g["var2"] = v2

        assert g.name in v1.groups
        assert g.name in v2.groups

    def test_delitem(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test", var1=v1, var2=v2)

        del g["var1"]

        assert "var1" not in g
        assert g.name not in v1.groups
        assert g.name in v2.groups

    def test_pop(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test", var1=v1, var2=v2)

        g.pop("var1")

        assert "var1" not in g
        assert g.name not in v1.groups
        assert g.name in v2.groups

    def test_clear(self) -> None:

        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test", var1=v1, var2=v2)
        g.clear()

        assert not g.data

        assert g.name not in v1.groups
        assert g.name not in v2.groups

    def test_update(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g = Group("test")
        g.update({"var1": v1, "var2": v2})

        assert g.name in v1.groups
        assert g.name in v2.groups

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

    def test_merge_operator(self) -> None:
        """
        Merging with the ``|`` operator does fails, because new groups need new names.
        """

        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g1 = Group("g1", var1=v1, var2=v2)
        g2 = Group("g2", nd1=n1)

        with pytest.raises(NotImplementedError):
            g1 | g2

    def test_update_inplace(self) -> None:
        """
        Merging inplace seems nice at first, but the new group membership of the
        members of group 2 is not updated in their respective .groups attribute.
        """

        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g1 = Group("g1", var1=v1, var2=v2)
        g2 = Group("g2", nd1=n1)

        g1 |= g2

        assert g1.name == "g1"  # name of g1 remains unchanged

        # group 1 gets updated
        assert len(g1) == 3
        assert "nd1" in g1

        # node gets updated
        assert len(n1.groups) == 2
        assert g1.name in n1.groups

    def test_merge_upacking(self) -> None:
        """
        Merging with unpacking works, but the new dict is not a group.
        """

        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g1 = Group("g1", var1=v1, var2=v2)
        g2 = Group("g2", nd1=n1)

        g3 = {**g1, **g2}

        with pytest.raises(AttributeError):
            g3.name  # type: ignore
        assert not isinstance(g3, Group)
        assert len(g3) == 3
        assert "nd1" in g3

    def test_merge_unpack_second(self) -> None:
        """This seems to be nice and explicit, things happen as expected."""
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g1 = Group("g1", var1=v1, var2=v2)
        g2 = Group("g2", nd1=n1)

        g3 = Group("g3", **g1, **g2)

        assert len(g3) == 3
        assert "nd1" in g3
        assert len(v1.groups) == 2

    def test_merge_update(self) -> None:
        """This seems to be nice and explicit, things happen as expected."""
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        n1 = Data(0.0)

        g1 = Group("g1", var1=v1, var2=v2)
        g2 = Group("g2", nd1=n1)

        g1.update(g2)

        assert g1.name == "g1"  # name of g1 remains unchanged

        # g1 gets updated
        assert len(g1) == 3
        assert "nd1" in g1

        # node gets updated
        assert g1.name in n1.groups
        assert len(n1.groups) == 2

        # members of g1 remain unchanged
        assert len(v1.groups) == 1

    def test_replace(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")

        g1 = Group("g1", var1=v1)
        g1["var1"] = v2

        # g1 gets updated
        assert len(g1) == 1

        assert v1 not in g1.values()
        assert v2 in g1.values()

        assert g1.name not in v1.groups
        assert not v1.groups

        assert g1.name in v2.groups

    def test_double_membership(self) -> None:
        """Each node can only be member of the same group once."""
        v1 = Var(0.0, name="v1")
        g1 = Group("g1", var1=v1)
        with pytest.raises(RuntimeError):
            g1["var2"] = v1

    def test_duplicate_name(self) -> None:
        """A node cannot be member of two groups with the same name."""
        v1 = Var(0.0, name="v1")
        g1 = Group("g1")
        g2 = Group("g1")

        g1["var1"] = v1
        with pytest.raises(RuntimeError):
            g2["var_name"] = v1

    def test_inconsistent_membership(self) -> None:
        """
        As a safeguard, there is an error if group membership is inconsistent.
        """
        # SCENARIO 1: Node loses information about group membershio
        v1 = Var(0.0, name="v1")
        g1 = Group("g1", var1=v1)

        # remove group from .groups dict without terminating group membership
        del v1.groups.data["g1"]

        assert "var1" in g1
        assert v1 in g1.values()
        assert "g1" not in v1.groups

        with pytest.raises(RuntimeError):
            g1["var1"] = v1

        # SCENARIO 2: Group loses information about node membershio
        v1 = Var(0.0, name="v1")
        g1 = Group("g1", var1=v1)

        # remove group from .groups dict without terminating group membership
        del g1.data["var1"]

        assert "var1" not in g1
        assert v1 not in g1.values()
        assert "g1" in v1.groups

        with pytest.raises(RuntimeError):
            g1["var1"] = v1


class TestGroupDict:
    def test_only_groups_allowed(self) -> None:
        v1 = Var(0.0, name="v1")
        with pytest.raises(ValueError, match="not a Group"):
            v1.groups["test"] = "string"  # type: ignore

    def test_add_group_under_wrong_name(self) -> None:
        """The key in GroupDict attribute must be equal to the group.name attribute."""
        v1 = Var(0.0, name="v1")
        g1 = Group("g1")
        g1.data["var1"] = v1  # workaround for membership without updating the node
        with pytest.raises(RuntimeError, match="not equal to group name"):
            v1.groups["g2"] = g1

    def test_add_group_without_membership_fails(self) -> None:
        v1 = Var(0.0, name="v1")
        g1 = Group("g1")
        with pytest.raises(RuntimeError, match="not a member"):
            v1.groups["g1"] = g1

    def test_remove_group(self) -> None:
        """
        When you remove a group from the GroupDict, membership data within the group
        is also updated, i.e. the node is no member of the group anymore.
        """
        v1 = Var(0.0, name="v1")
        g1 = Group("g1", var1=v1)

        del v1.groups["g1"]
        assert "var1" not in g1
        assert "g1" not in v1.groups

    def test_replace_fails(self) -> None:
        """You cannot simply replace a group."""
        v1 = Var(0.0, name="v1")
        Group("g1", var1=v1)

        with pytest.raises(
            RuntimeError, match="already a member of a group with the name"
        ):
            v1.groups["g1"] = Group("g1")
