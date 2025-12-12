import liesel.model as lsl
import liesel.experimental.optim as opt


def test_vdist_float64():
    x = lsl.Var.new_obs(
        0.,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0., 1.)
    
    assert not q.to_float32


def test_compositevdist_float64():
    x = lsl.Var.new_obs(
        0.,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0., 1.)
    
    vi_dist = opt.CompositeVDist(q).build()

    assert not vi_dist.to_float32

