def layer_limits(ysz=200.0, tgo=1.0, bond=150.0, sub=1000.0):
    # um
    y0 = 0.0
    y1 = y0 + sub
    y2 = y1 + bond
    y3 = y2 + tgo
    y4 = y3 + ysz
    return y0, y1, y2, y3, y4
