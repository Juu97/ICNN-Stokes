from firedrake import *

def dudx(x, y):
    return 2 * x * y * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y))


def dudy(x, y):
    return - 8 * y * y * cos(x * x - y * y) + 10 * y * y * cos(x * x + y * y) + 4 * sin(x * x - y * y) + 5 * sin(
        x * x + y * y)


def dvdx(x, y):
    return 8 * x * x * cos(x * x - y * y) - 10 * x * x * cos(x * x + y * y) + 4 * sin(x * x - y * y) - 5 * sin(
        x * x + y * y)


def dvdy(x, y):
    return - 2 * x * y * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y))


def d11(x, y):
    return dudx(x, y)


def d12(x, y):
    return 0.5 * dvdx(x, y) + 0.5 * dudy(x, y)


def d22(x, y):
    return dvdy(x, y)


def dd11dy(x, y):
    return 2 * x * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y) + 2 * y * y * (
            4 * sin(x * x - y * y) - 5 * sin(x * x + y * y)))


def dd11dx(x, y):
    return 2 * y * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y) - 2 * x * x * (
            4 * sin(x * x - y * y) + 5 * sin(x * x + y * y)))


def dd12dy1(x, y):
    return 0.5 * (2 * y * (-4 * cos(x * x - y * y) - 5 * cos(x * x + y * y) + 2 * x * x * (
            4 * sin(x * x - y * y) + 5 * sin(x * x + y * y))))


def dd12dy2(x, y):
    return 0.5 * (-24 * y * cos(x * x - y * y) + 30 * y * cos(x * x + y * y) - 16 * y * y * y * sin(
        x * x - y * y) - 20 * y * y * y * sin(x * x + y * y))


def dd12dy(x, y):
    return dd12dy1(x, y) + dd12dy2(x, y)


def dd12dx1(x, y):
    return 0.5 * (24 * x * cos(x * x - y * y) - 30 * x * cos(x * x + y * y) + 4 * x * x * x * (
            -4 * sin(x * x - y * y) + 5 * sin(x * x + y * y)))


def dd12dx2(x, y):
    return 0.5 * (2 * x * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y) + 2 * y * y * (
            4 * sin(x * x - y * y) - 5 * sin(x * x + y * y))))


def dd12dx(x, y):
    return dd12dx1(x, y) + dd12dx2(x, y)


def dd22dy(x, y):
    return -2 * x * (4 * cos(x * x - y * y) + 5 * cos(x * x + y * y) + 2 * y * y * (
            4 * sin(x * x - y * y) - 5 * sin(x * x + y * y)))


def dd22dx(x, y):
    return 2 * y * (-4 * cos(x * x - y * y) - 5 * cos(x * x + y * y) + 2 * x * x * (
            4 * sin(x * x - y * y) + 5 * sin(x * x + y * y)))


def dpdx(x, y):
    return cos(x + y)


def dpdy(x, y):
    return cos(x + y)


def dnudx(x, y, r=1.6):
    n = r - 1
    nu0 = 2.0
    nuinf = 0.0

    return (nu0 - nuinf) * (n - 1) / 2 * (
            1 + (2 * dudx(x, y) * dudx(x, y) + 4 * d12(x, y) * d12(x, y) + 2 * d22(x, y) * d22(x, y))) ** (
                   (n - 3) / 2) * (
                   4 * dudx(x, y) * dd11dx(x, y) + 8 * d12(x, y) * dd12dx(x, y) + 4 * dvdy(x, y) * dd22dx(x, y))


def dnudy(x, y, r=1.6):
    n = r - 1
    kt = 1.0
    nu0 = 2.0
    nuinf = 0.0

    return (nu0 - nuinf) * (n - 1) / 2 * (
            1 + (2 * dudx(x, y) * dudx(x, y) + 4 * d12(x, y) * d12(x, y) + 2 * d22(x, y) * d22(x, y))) ** (
                   (n - 3) / 2) * (
                   4 * dudx(x, y) * dd11dy(x, y) + 8 * d12(x, y) * dd12dy(x, y) + 4 * dvdy(x, y) * dd22dy(x, y))

def u_ex(x, y,r=1.6):
    return 5 * y * sin(x * x + y * y) + 4 * y * sin(x * x - y * y), \
          -5 * x * sin(x * x + y * y) + 4 * x * sin(x * x - y * y)

def p_ex(x, y, r=1.6):
    return sin(x + y)

def nu(x, y, r=1.6):
    mu_inf = Constant(0.0)  # Viscosity at infinite shear rate
    mu_0 = Constant(2)  # Viscosity at zero shear rate
    lam = Constant(1)  # Time constant
    n = r - 1  # Dimensionless parameter

    # Compute effective viscosity using Carreau law, here you have lambda*||eps()||^2 => lambda = 2
    mu_eff = mu_inf + (mu_0 - mu_inf) * (
            1 + (2 * dudx(x, y) * dudx(x, y) + 4 * d12(x, y) * d12(x, y) + 2 * d22(x, y) * d22(x, y))) ** (
                     (n - 1) / 2)
    return mu_eff

def fx_fy(x, y,r=1.6):
    return -2.0 * dnudx(x, y,r) * d11(x, y) - 2.0 * nu(x, y,r) * dd11dx(x, y) - 2.0 * dnudy(x, y,r) * d12(x, y) - 2.0 * nu(x,
                                                                                                                     y,r) * dd12dy(
        x, y) + dpdx(x, y), \
           -2.0 * dnudx(x, y,r) * d12(x, y) - 2.0 * nu(x, y,r) * dd12dx(x, y) - 2.0 * dnudy(x, y,r) * d22(x, y) - 2.0 * nu(x,
                                                                                                                     y,r) * dd22dy(
               x, y) + dpdy(x, y)