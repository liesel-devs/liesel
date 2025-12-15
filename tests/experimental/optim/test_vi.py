import liesel.experimental.optim as opt
import liesel.model as lsl


def test_vdist_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0)

    assert not q.p.to_float32


def test_compositevdist_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0)

    vi_dist = opt.CompositeVDist(q).build()

    assert not vi_dist._to_float32()
