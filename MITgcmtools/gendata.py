#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data generating methods for the MITgcm.

Examples for the gendata cli tool:

Example:
Generate turbulence : gendata -N 120x120x120 -o Uinit.bin TurbulentVel U0=0.01 zmode=4

Example 2:
Multiple presets can be used (by addition of the generated data) as (generate gaussian jet with checkerboard like perturbation)
gendata -N 120 -o Uinit.bin UJetGauss U=1 +UCheckerboard U=0.1

"""
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from warnings import warn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import xarray as xr
from .common import baseArgParser, _sizeParser, _listParser, COLORS, parse_kwarg

PRESETS = []
OPERATORS = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "_": lambda a, b: np.where(a > b, a, b),
}


def _get_np_from_xr(arr):
    if isinstance(arr, xr.DataArray):
        if {'x', 'y'} == set(arr.dims):
            arr = arr.transpose('x', 'y')
        elif {'x', 'y', 'z'} == set(arr.dims):
            arr = arr.transpose('x', 'y', 'z')
        return arr.values
    return arr


def writeIEEEbin(filename, arr, precision=64):
    """Write a numpy array to a IEEE (Big-Endian) binary file."""
    arr = _get_np_from_xr(arr)
    # Python is in C-layout (X, Y, Z), so we have to flip to F-layout (Z, Y, X)
    if arr.ndim == 2: 
        arr = arr.T
    elif arr.ndim == 3:
        arr = arr.swapaxes(0, 2)
    # Precision is in bits, the dtype is in Bytes. > : IEEE, f : float, 4/8 : precision
    print(f"Writing data of shape {arr.shape} to file {filename} with precision of {precision} bits")
    arr.astype(">f{:d}".format(precision // 8)).tofile(str(filename))


def write_concatenate(filename, *arrs, precision=64):
    """Write multiple numpy arrays to a single file."""
    arrays = [_get_np_from_xr(arr) for arr in arrs]
    conc = np.stack(arrays, axis=-1)
    writeIEEEbin(filename, conc, precision=precision)

def readIEEEbin(filename, shape, precision=64):
    """Read a IEEE (Big-Endian) binary file.

    Arguments:
        filename -- Filename to read
        shape -- Shape of the array

    Keyword arguments:
        precision -- number of bits for each float (default: {64} (8 B))
    """
    arr = np.fromfile(
        str(filename), dtype=np.dtype(">f{:d}".format(precision // 8))
    ).reshape(shape[::-1])
    if arr.ndim == 2:
        return arr.T
    if arr.ndim == 3:
        return arr.swapaxes(0, 2)
    return arr


def get_grid(size, mode="", levels=None):
    """Create a meshgrid with the correct normalized coordinates.

    Arguments:
        size -- The nx, ny, nz size of the domain

    Keyword arguments:
        mode -- The C-grid node of the grid (default: {''}, tracer point)
                (other choices : 'u', 'v', 'w'))
        levels -- A list of the z-levels, If none, a normalised vector is returned.
                  (default: {None})
    """
    # Create the coordinate vector on the cell center or at u/v/w point following preset.grid
    coords = [
        np.arange(Ni) / Ni if mode == mo else (np.arange(Ni) + 0.5) / Ni
        for Ni, mo in zip(size[: (-1 if levels is not None else None)], ("u", "v", "w"))
    ]
    # If levels is given, treat the last coord specially
    if levels is not None:
        if mode == "w":
            coords = coords + [(np.array(levels) - levels[0]) / levels[-1]]
        else:
            coords = coords + [
                (np.array(levels) - np.diff(np.concatenate(([0], levels))) / 2) / levels[-1]
            ]
    return np.meshgrid(*coords, indexing="ij")


def fill_datafield(field, data, folder="."):
    """Fill a datafiled in the "data" namelist file with the horizontal mean of an array

    Useful for setting tRef or sRef from a generated field.
    The field must already exist in "data".

    Arguments:
        field -- Name of the field to fill
        data -- Array to use

    Keyword arguments:
        folder -- (default: {'.'})
    """
    folder = Path(folder).resolve()
    with (folder / "data.tempnew.gendata").open("w") as newdataf:
        with (folder / "data").open("r") as olddataf:
            for line in olddataf:
                if line.strip().casefold().startswith(field.casefold()):
                    newdataf.write(
                        " {}={},\n".format(
                            field,
                            ",".join(
                                [
                                    "{:.2f}".format(val)
                                    for val in np.mean(data, axis=(0, 1))
                                ]
                            ),
                        )
                    )
                else:
                    newdataf.write(line)
    (folder / "data.tempnew.gendata").rename(folder / "data")
    # print('Vertical structure written to file "data".')


def get_preset(name, ndim=None, timepattern=False):
    """Return the preset "name" from PRESETS, case insensitive.

    Keyword arguments:
        ndim -- Either 2 or 3, return the specific version of the preset (default: {None})
        timepattern -- Return a timepattern with this name (default: {False})

    Raises a {ValueError} if it doesn't find the corresponding preset.
    """
    name = name.casefold()
    for preset in PRESETS:
        if (
            preset.name.casefold() == name
            and (ndim is None or preset.ndim == ndim)
            and (timepattern == isinstance(preset, TimePatternPreset))
        ):
            return preset
    raise ValueError(
        "Invalid preset name {} for data of {} dims.".format(name, ndim or "any")
    )


class Preset:
    """Preset object, callable, with description (1 str + about potential kwargs) and a fancy docstring. """

    def __init__(self, name, func, desc=None, kwargs=None):
        """
        Initialize the data preset.

        Arguments:
            name -- Name of the data preset in one word (no spaces, no -/*_ etc)
            func -- A function taking the 3D meshgrids X, Y[, Z] of the normalised coords [0, 1] and some kwargs

        Keyword arguments:
            desc -- A short one-line description of the preset. (default: {None})
            kwargs -- A dict of all possible extra arguments. Key is the name and value is a short description with the default in ()
                      (default: {None})
        """
        self.name = name
        self.kwargs = kwargs or {}
        self.func = func
        self.desc = desc
        self.__doc__ = (
            (
                COLORS.YELLOW
                + "{classname} '{name}'."
                + COLORS.ENDC
                + "\n\t"
                + COLORS.BLUE
                + "{desc}"
                + COLORS.ENDC
                + "\n\t"
                + COLORS.YELLOW
                + "Args : Desc (default):"
                + COLORS.ENDC
            ).format(
                classname=self.__class__.__name__,
                name=name,
                desc=desc or "No description available",
            )
            + "\n\t"
            + "-" * 22
            + "\n\t"
            + "\n\t".join(
                [
                    (COLORS.GREEN + COLORS.BOLD + "{}" + COLORS.ENDC + "{}").format(
                        nam, doc
                    )
                    for nam, doc in self.kwargs.items()
                ]
            )
        )

    def __call__(self, *args, **kwargs):
        """Shortcut to the func callable."""
        return self.func(*args, **kwargs)

    def __repr__(self):
        return (
            "<{} {} | ".format(self.__class__.__name__, self.name)
            + ", ".join(self.kwargs.keys())
            + ">"
        )

    def __str__(self):
        return self.__doc__


class DataPreset(Preset):
    """Initial data preset, taking the 3D meshgrids for X, Y, Z and returning an initial data cube."""

    ndim = 3

    def __init__(self, name, func, desc=None, kwargs=None, grid="t"):
        """
        Same a parent + grid : The grid on which the data is generated, one of 't', 'u', 'v', 'w'
        """
        super().__init__(
            name, func, desc=(desc or "") + "(On grid {})".format(grid), kwargs=kwargs
        )
        self.grid = grid


class SurfacePreset(DataPreset):
    """Specifc case of DataPreset for 2D surface data."""

    ndim = 2

    pass


class TimePatternPreset(Preset):
    """Time patterns take generated data and repeat it along a new axis following some transformation."""

    def __call__(self, data, **kwargs):
        return np.moveaxis(self.func(data, **kwargs), 0, -1)


# #################################################################### #
# #################################################################### #
# ####################  Preset definitions  ########################## #


def _simplebathy(X, Y, H=5000, slices=None):
    """Create a bathymetry of depth H with walls at slices."""
    bathy = -H * np.ones(X.shape)
    for wall in slices or []:
        bathy[wall] = 0
    return bathy


def _simplebathyfun(*slices):
    """Generate a bathymetry function from slices where walls should be."""
    return lambda X, Y, H=5000: _simplebathy(X, Y, H=H, slices=slices)


# Bathys
PRESETS.append(
    SurfacePreset(
        "Box",
        lambda X, Y, H=5000: _simplebathy(
            X, Y, H=H, slices=[[slice(None), -1], [-1, slice(None)]]
        ),
        kwargs={"H": "Depth of the model (m) (5000)"},
        desc="Simple box bathymetry.",
    )
)
PRESETS.append(
    SurfacePreset(
        "XChannel",
        lambda X, Y, H=5000: _simplebathy(X, Y, H=H, slices=[[slice(None), -1]]),
        kwargs={"H": "Depth of the model (m) (5000)"},
        desc="Simple zonal channel bathymetry.",
    )
)
PRESETS.append(
    SurfacePreset(
        "YChannel",
        lambda X, Y, H=5000: _simplebathy(X, Y, H=H, slices=[[-1, slice(None)]]),
        kwargs={"H": "Depth of the model (m) (5000)"},
        desc="Simple meridional channel bathymetry.",
    )
)
PRESETS.append(
    SurfacePreset(
        "Wall",
        lambda X, Y, H=5000, x=slice(None), y=slice(None): _simplebathy(
            X, Y, H=H, slices=[[x, y]]
        ),
        kwargs={
            "x": "X Position of the wall (cell index) (all x)",
            "y": "Y Position of the wall (cell index) (all y)",
            "H": "Depth of the model (m) (5000)",
        },
        desc="Simple wall bathymetry. Specify one of x or y.",
    )
)

# Surface
PRESETS.append(
    SurfacePreset(
        "GaussianNoise",
        lambda X, Y, V0=100, dV=1: V0 + dV * (np.random.random(X.shape) - 0.5),
        kwargs={"V0": "Base (mean) value (100)", "dV": "Noise amplitude (1)"},
        desc="Constant valued surface with random noise.",
    )
)


def _leads(X, Y, Vs=[0, 1, 0], Ws=0.5, Cs=0.5, dw=0.05, zonal=False):
    if np.isscalar(Ws):
        Ws = (Ws,)
        Cs = (Cs,)
    if len(Vs) == 2 * len(Ws):
        Vs.append(Vs[0])
    ys = [s for w, c in zip(Ws, Cs) for s in [c - (w / 2), c + (w / 2)]]
    return (
        np.sum(
            (np.tanh(((Y if zonal else X) - ys[i]) / (0.275 * dw)) + 1)
            / 2
            * (Vs[i + 1] - Vs[i])
            for i in range(len(ys))
        )
        + Vs[0]
    )


PRESETS.append(
    SurfacePreset(
        "Leads",
        _leads,
        kwargs={
            "Vs": "Values at all stages of the lead(s) (left, middle, right) (0, 1, 0)",
            "Ws": "Widths of all leads, scalar if only one. (0.5)",
            "Cs": "Center coordinates of all leads, scalar if only one. (0.5)",
            "dw": "Thickness of 95\% of the hyperbolic tangent transitions (0.05)",
            "zonal": "Wheter the leads are zonal (E-W) or not (N-S) (False)",
        },
        desc="Step-like surface for representing leads of other bands.",
    )
)


def _oneLead(X, Y, V=0.2, W=0.2, top=0.1, pad=0, zonal=False, mode="linear"):
    c = Y if zonal else X
    N, l, ax = c.shape[0], c, int(zonal)
    L = np.roll(l * 2 - 1, N // 2, axis=ax) * -abs(np.tanh((l - 0.5) / (0.29 * W)))
    if mode == "plateau":
        Lmax = L[c > (0.5 + top)][0]
        L[L > Lmax] = Lmax
        Lmin = L[c < (0.5 - top)][-1]
        L[L < Lmin] = Lmin
    if pad > 0:
        L = np.pad(
            L,
            [(0, 0)] * ax + [(pad, pad)] + [(0, 0)] * int(not zonal),
            mode="constant",
            constant_values=0,
        )
    return V * L / L.max()


PRESETS.append(
    SurfacePreset(
        "OneLead",
        _oneLead,
        kwargs={
            "V": "Maximal amplitude of τ left (under) of the lead (0.2)",
            "W": "Width of the lead (0.2)",
            "S": "Width of the enveloppe (0.5)",
            "a": "Corrective width factor (0.5)",
            "top": "Size of the plateau (0.1)",
            "zonal": "Whether the lead is zonal or not (False)",
            "mode": "What mode to use : linear the convectionnal, tanh with steeper descent on the sides or plateau.",
        },
        desc="Linear stress with jump in the middle.",
    )
)

# Deprecated
PRESETS.append(
    SurfacePreset(
        "Lead",
        lambda X, Y, V0=100, w=0.5, dw=0.025, y0=0.5: V0
        * 0.5
        * (
            1
            + np.where(X < 0.5, -1, 1)
            * np.tanh((y0 + np.where(X < 0.5, -1, 1) * (w / 2) - X) / dw)
        ),
        kwargs={
            "V0": "Amplitude of in the lead (100)",
            "w": "Width of the lead (0.5)",
            "dw": "tanh ramp width. (0.025)",
            "y0": "Lead center (0.5)",
        },
        desc="DEPRECATED (use Leads) Band along the X direction with hyperbolic tangent transition on the sides.",
    )
)


PRESETS.append(
    SurfacePreset(
        "Sine",
        lambda X, Y, axis="X", V0=1, n=1: V0
        * np.sin((X if axis == "X" else Y) * 2 * np.pi * n),
        kwargs={
            "axis": "Direction of the sine variation (X or Y)",
            "V0": "Amplitude of the sine",
            "n": "Number of waves",
        },
        desc="Sinusoidal surface",
    )
)

# Vels
PRESETS.append(
    DataPreset(
        "UJetGauss",
        lambda X, Y, Z, U=1, mu=0.5, sig=0.05: U * np.exp(-((Y - mu) / sig) ** 2),
        grid="u",
        kwargs={
            "U": "Maximal speed of the jet (1)",
            "mu": "Y-position of the jet max (0.5)",
            "sig": "Standard deviation of the gaussian jet (0.05)",
        },
    )
)
PRESETS.append(
    DataPreset(
        "VJetGauss",
        lambda X, Y, Z, V=1, mu=0.5, sig=0.05: V * np.exp(-((X - mu) / sig) ** 2),
        grid="v",
        kwargs={
            "V": "Maximal speed of the jet (1)",
            "mu": "X-position of the jet max (0.5)",
            "sig": "Standard deviation of the gaussian jet (0.05)",
        },
    )
)
PRESETS.append(
    DataPreset(
        "UCheckerboard",
        lambda X, Y, Z, U=0.01, n=8: -U
        * np.sin(2 * n * np.pi * X)
        * np.cos(2 * n * np.pi * Y),
        grid="u",
        kwargs={
            "U": "Amplitude of checkerboard perturbation (0.01)",
            "n": "Number of cycles per side (8)",
        },
    )
)

PRESETS.append(
    DataPreset(
        "VCheckerboard",
        lambda X, Y, Z, V=0.01, n=8: V
        * np.cos(2 * n * np.pi * X)
        * np.sin(2 * n * np.pi * Y),
        grid="v",
        kwargs={
            "V": "Amplitude of checkerboard perturbation (0.01)",
            "n": "Number of cycles per side (8)",
        },
    )
)


def _turbulent_injection(X, Y, Z, kmin=3, kmax=5, U0=0.01, zmode=4):
    if np.isscalar(kmin):
        lmin = kmin
        lmax = kmax
    else:
        kmin, lmin = kmin
        kmax, lmax = kmax
    Fu = np.zeros((X.shape[0], X.shape[1], X.shape[2] // 2 + 1), dtype=np.complex)
    for k in range(-kmax, kmax + 1):
        for l in range(-lmax, lmax + 1):
            if (k / kmin) ** 2 + (l / lmin) ** 2 >= 1 and (k / kmax) ** 2 + (
                l / lmax
            ) ** 2 <= 1:
                Fu[k, l, 0] = np.random.rand() * np.exp(
                    1j * 2 * np.random.rand() * np.pi
                )
    U = np.fft.irfftn(Fu) * np.cos(2 * np.pi * zmode * Z)
    return U0 * U / np.mean(np.abs(U))


PRESETS.append(
    DataPreset(
        "TurbulentVel",
        _turbulent_injection,
        desc="Turbulent field from random horizontal fourier-space constrained between kmin and kmax. Single vertical mode.",
        kwargs={
            "kmin": "Lower boundary of kH-space energy injection (can be a tuple (kx, ky)) (3)",
            "kmax": "Upper boundary of kH-space energy injection (can be a tuple (kx, ky)) (5)",
            "U0": "Velocities are renormalised so that the mean of the absolute U is U0. (m/s) (0.01)",
            "zmode": "Mode in z (cos(2*pi*Z*zmode)), Z is given in [0, 1]. (4)",
        },
    )
)

# Temp
PRESETS.append(
    DataPreset(
        "SurfaceAnomaly",
        lambda X, Y, Z, A=2, sH=0.03, sV=0.5, x0=0.5, y0=0.5, z0=0: A
        * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / sH ** 2) - ((Z - z0) / sV) ** 2),
        kwargs={
            "A": "Amplitude of the anomaly (2)",
            "sH": "Horizontal standard deviation of the anomaly (0.03)",
            "sV": "Vertical standard deviatio of the anomaly (0.5)",
            "x0": "X-position of the anomaly (0.5)",
            "y0": "Y-position of the anomaly (0.5)",
            "z0": "Z-position of the anomaly (0)",
        },
    )
)


def _baroclinic_anomaly(X, Y, Z, A=2, sH=0.03, x0=0.5, y0=0.5, a1=1, a2=0, a3=0):
    zcomp = np.zeros(Z.shape)
    for n, a in enumerate([a1, a2, a3]):
        zcomp = zcomp + a * np.sin((n + 1) * np.pi * Z)

    return zcomp * A * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / sH ** 2))


PRESETS.append(
    DataPreset(
        "BaroclinicAnomaly",
        _baroclinic_anomaly,
        kwargs={
            "A": "Amplitude of the anomaly (2)",
            "sH": "Horizontal standard deviation of the anomaly (0.03)",
            "x0": "X-position of the anomaly (0.5)",
            "y0": "Y-position of the anomaly (0.5)",
            "a1": "Amplitude of the first baroclinic mode (1)",
            "a2": "Amplitude of the second baroclinic mode (0)",
            "a3": "Amplitude of the third baroclinic mode (0)",
        },
    )
)


PRESETS.append(
    DataPreset(
        "MixedLayerLinearStrat",
        lambda X, Y, Z, mld=0.2, pml=0, ddz=0.05: pml
        + np.where(Z <= mld, 0, ddz * (Z - mld) * Z.shape[-1]),
        kwargs={
            "mld": "Mixed layer depth [0, 1) (0.5)",
            "pml": "Constant value for the mixed layer. (5)",
            "pbot": "Value at the bottom. (None)",
            "ddz": "Linear stratification coeff (temp diff if mld == 0) (None)",
            "smwin": "Length of the running mean window (fraction of profile) (0.166 i.e. 1/6)",
        },
        desc="Profile with constant valued mixed layer at the top and linear stratification at the bottom. (One of Tbot or dTdz must be given.)",
    )
)


def _mixed_layer_exp_strat(X, Y, Z, mld=0.2, ddz=[0.05, 0.005], pml=28):
    alpha = np.log(ddz[1] / ddz[0]) / (Z[0, 0, -1] - mld)
    dP = np.where(Z <= mld, 0, ddz[0] * np.exp(alpha * (Z - mld)))
    return np.cumsum(dP, axis=-1) + pml


PRESETS.append(
    DataPreset(
        "MixedLayerExpStrat",
        _mixed_layer_exp_strat,
        kwargs={
            "mld": "Mixed layer depth [0, 1), (0.2)",
            "pml": "Constant value for the mixed layer. (28)",
            "ddz": "Stratification at the base of the ML and at the bottom (0.05, 0.005)",
        },
        desc="Profile with constant valued mixed layer at the top and exponential stratification in the rest.",
    )
)


def _from_profile(X, Y, Z, file=None, var="profile", dz=None, Zbot=None, V0=0):
    if file.endswith("mat"):
        data = sio.loadmat(file, squeeze_me=True)
    elif file.endswith("npy") or file.endswith("npz"):
        data = np.load(file)
    else:
        raise NotImplementedError("Filetype not yet understood.")

    if dz is None and Zbot is None and data[var].shape[0] == Z.shape[-1]:
        # Same shape, no dz or Zbot, simply copy.
        profile = np.ones(Z.shape) * V0
        profile[..., :] = data[var][np.newaxis, np.newaxis, :]
        return profile

    if "Z" in data:  # The z-coordinate vector of the profile is given
        Zi = data["Z"]
        if Zi[1] < 0:
            Zi = -Zi
    elif "dz" in data:  # The z-coordinate vector is equally spaced on given dz
        Zi = np.arange(0.5, 0.6 + data[var].shape[0]) * data["dz"]
    else:
        Zi = np.arange(0.5, 0.6 + data[var].shape[0])
        warn("The data file contains neither dz or Z. dz = 1 is assumed.")

    if dz is None:
        if Zbot is None:
            Zbot = Zi[-1]
        Zp = Z * Zbot
    else:
        if Zbot is not None:
            raise ValueError("Cannot give both Zbot and dz.")
        Zp = Z * dz * Z.shape[-1]

    return interp1d(Zi, data[var], bounds_error=False, fill_value=V0)(Zp)


PRESETS.append(
    DataPreset(
        "FromProfile",
        _from_profile,
        kwargs={
            "file": "Filename where the profile is stored. Must be npy, npz or mat. ()",
            "var": 'For npz and mat, name of the profile variable ("profile")',
            "dz": "If specified, cell depth to reinterpolate the profile data. ()",
            "Zbot": "Depth at the bottom of the model ()",
            "V0": "Padding value if the profile is too short.",
        },
        desc="3D field set from 1D z-profile file. If the profile has a different size as the domain, it is reinterpolated.",
    )
)


def _uniform_region(
    X, Y, Z, cellsize=[1, 1, 1], lims=[0, None, 0, None, 0, None], vals=[1, 0]
):
    lims = [
        int(lim / cellsize[i // 2]) if lim is not None else None
        for i, lim in enumerate(lims)
    ]
    ptracer = np.ones(X.shape) * vals[1]
    ptracer[lims[0] : lims[1], lims[2] : lims[3], lims[4] : lims[5]] = vals[0]
    return ptracer


PRESETS.append(
    DataPreset(
        "UniformRegion",
        _uniform_region,
        kwargs={
            "cellsize": "Size of the cell : dx,dy,dz, (1,1,1)",
            "lims": "Limits of the region : x1, x2, y1, y2, z1, z2. None means end. (0,None,0,None,0,None)",
            "vals": "Values to use in and out the region : vIn, vOut (1,0)",
        },
        desc="Creates a field with one region set at one value. If cellsize is not given, lims are interpreted as indexes.",
    )
)

PRESETS.append(
    DataPreset(
        "GaussianNoise",
        lambda X, Y, Z, V0=100, dV=1: V0 + dV * (np.random.random(X.shape) - 0.5),
        kwargs={"V0": "Base (mean) value (100)", "dV": "Noise amplitude (1)"},
        desc="Constant valued volume with random noise.",
    )
)

PRESETS.append(
    DataPreset(
        "GaussianProfile",
        lambda X, Y, Z, mu=0.2, a=1, sig=0.033: a * np.exp(-((Z - mu) / sig) ** 2),
        kwargs={
            "mu": "Z-Position of the gaussian bump. (0.2)",
            "a": "Amplitude of the bump.(1)",
            "sig": "Standard deviation of the gaussian bump. (0.033)",
        },
        desc="Creates a 3D field with a gaussian profile.",
    )
)

PRESETS.append(
    DataPreset(
        "TanhTransitionX",
        lambda X, Y, Z, x0=0.5, V0=0, V1=1, dX=0.05: (
            np.tanh((X - x0) / (0.275 * dX)) + 1
        )
        / 2
        * (V1 - V0)
        + V0,
        kwargs={
            "x0": "Location of the transition's center (0.5)",
            "V0": "Value east (right) of the transition (0)",
            "V1": "Value west (left) of the transition (1)",
            "dX": "Thickness of the transition (0.05)",
        },
        desc="Bi-valued 3D field with an hyperbolic tangent transition in X",
    )
)


PRESETS.append(
    DataPreset(
        "TanhTransitionY",
        lambda X, Y, Z, y0=0.5, V0=0, V1=1, dY=0.05: (
            np.tanh((Y - y0) / (0.275 * dY)) + 1
        )
        / 2
        * (V1 - V0)
        + V0,
        kwargs={
            "y0": "Location of the transition's center (0.7)",
            "V0": "Value north of the transition (0)",
            "V1": "Value south of the transition (1)",
            "dZ": "Thickness of the transition (0.05)",
        },
        desc="Bi-valued 3D field with an hyperbolic tangent transition in Y",
    )
)


PRESETS.append(
    DataPreset(
        "TanhTransition",
        lambda X, Y, Z, z0=0.7, V0=0, V1=1, dZ=0.05: (
            np.tanh((Z - z0) / (0.275 * dZ)) + 1
        )
        / 2
        * (V1 - V0)
        + V0,
        kwargs={
            "z0": "Depth of the transition's center (0.7)",
            "V0": "Value above the transition (0)",
            "V1": "Value below the transition (1)",
            "dZ": "Thickness of the transition (0.05)",
        },
        desc="Bi-valued 3D field with an hyperbolic tangent transition in Z",
    )
)


PRESETS.append(
    TimePatternPreset(
        "XCycle",
        lambda data, steps=-1, first=0: np.array(
            [
                np.concatenate((data[i:], data[:i]), axis=0)
                for i in np.arange(
                    first, first + np.sign(-steps) * data.shape[0], -steps
                )
            ]
        ),
        kwargs={
            "steps": "Steps of the cycle, >0 is Eastward (1)",
            "first": "First x-index where to cut (0)",
        },
        desc="Cycles the data in the x direction, new dimensions is shape[0] // steps",
    )
)

PRESETS.append(
    TimePatternPreset(
        "YCycle",
        lambda data, steps=-1, first=0: np.array(
            [
                np.concatenate((data[:, i:, ...], data[:, :i, ...]), axis=1)
                for i in np.arange(
                    first, first + np.sign(-steps) * data.shape[0], -steps
                )
            ]
        ),
        kwargs={
            "steps": "Steps of the cycle, >0 is Northward (1)",
            "first": "First y-index where to cut (0)",
        },
        desc="Cycles the data in the y direction, new dimensions is shape[0] // steps",
    )
)


def _base_timepattern(data, pattern):
    return (
        np.array(pattern)[(slice(None, None),) + data.ndim * (np.newaxis,)]
        * data[np.newaxis, ...]
    )


PRESETS.append(
    TimePatternPreset(
        "Linear",
        lambda data, C=0, D=1: _base_timepattern(data, [C, D, C, D]),
        kwargs={
            "C": "Coefficient of data at the beginning (0)",
            "D": "Coefficient at the end (1)",
        },
        desc="Linear oscillation (sawtooth wave) between C*data and D*data. MITgcm uses linear interpolation already. A ramp is (1, -1), with period *= 2",
    )
)

PRESETS.append(
    TimePatternPreset(
        "Cosine",
        lambda data, N=10: _base_timepattern(np.cos(2 * np.pi * np.arange(N) / N)),
        kwargs={"N": "Number of points in a cycle."},
        desc="Cosine oscillation of data in N points.",
    )
)


def _stormy(data, steps, U=0, V=0):
    data_storm = _base_timepattern(
        data,
        np.concatenate(
            (
                np.linspace(0, 1, steps[0], endpoint=False),
                np.ones((steps[1],)),
                np.linspace(1, 0, steps[2], endpoint=False),
                np.zeros((steps[3],)),
            )
        ),
    )
    if (U + V) != 0:
        for i in range(1, data_storm.shape[0]):
            data_storm[i] = np.roll(data_storm[i], i * int(U), axis=0)
            data_storm[i] = np.roll(data_storm[i], i * int(V), axis=1)
    return data_storm


PRESETS.append(
    TimePatternPreset(
        "Stormy",
        _stormy,
        kwargs={
            "steps": "Number of steps for the ramp up, storm, ramp down, quiet end.",
            "U": "Speed in gridcell / timestep in X (array is rolled)",
            "V": "Speed in gridcell / timestep in Y (array is rolled)",
        },
        desc="Multiplies the data so that it ramps up, stays constant, ramps down and stays null, options to cycle in X and Y.",
    )
)


def main():
    """Entry-point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='CLI interface for initial data generation for the MITgcm.\n Use "gendata list" to list all available presets and their options.',
        parents=[baseArgParser],
    )
    parser.add_argument(
        "-N",
        "--size",
        help="Size of the simulation as NXxNYxNR. If only one number, a cube is assumed.",
        type=_sizeParser,
        default=[120, 120, 120],
    )
    parser.add_argument(
        "-p",
        "--precision",
        help="Floating point precision to write the values. Default : 64",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-Z",
        "--levels",
        help=(
            "Specify the z-level coordinates for non-uniform grids. "
            "If shorter than size[-1] or non-monotonically-increasing, assumes level heights. "
            "If shorter, fills the beginning of the vector with the first given value."
        ),
        type=_listParser(float),
    )
    parser.add_argument(
        "--fill",
        help="Writes the mean vertical profile to given field in file data (current folder) (tref or sref).",
    )
    parser.add_argument(
        "--npy",
        help="Writes the array to a numpy npy file with the same name.",
        action="store_true",
    )
    parser.add_argument("preset", help="Name of the preset to use.")
    parser.add_argument(
        "kwargs",
        help='Extra keyword arguments specific to the preset, listed as "arg=val". Or additional presets to be added (as Xpresetname, where X is an operator in {}).'.format(
            ",".join(OPERATORS.keys())
        ),
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.preset == "list":
        print(*PRESETS, sep="\n\n")
        return

    if args.out is None:
        args.out = args.preset

    presets = [(get_preset(args.preset, len(args.size)), {}, None)]
    timepatterns = []
    kwargs = presets[0][1]
    for argStr in args.kwargs:
        if argStr[0] in OPERATORS:
            presets.append(
                (get_preset(argStr[1:], len(args.size)), {}, OPERATORS[argStr[0]])
            )
            kwargs = presets[-1][1]
        elif argStr[0] == "~":
            timepatterns.append((get_preset(argStr[1:], timepattern=True), {}))
            kwargs = timepatterns[-1][1]
        else:
            key, val = argStr.split("=")
            kwargs[key] = parse_kwarg(val)

    levels = args.levels
    if levels:
        if len(args.levels) < args.size[-1]:
            levels = np.array(
                [args.levels[0]] * (args.size[-1] - len(args.levels)) + args.levels
            ).cumsum()
        elif np.all(np.diff(args.levels) > 0):
            levels = np.array(args.levels).cumsum()

    print(
        "Generating initial data for preset(s) {}.".format(
            ", ".join([p.name for p, k, o in presets])
        )
    )

    grid = get_grid(args.size, mode=presets[0][0].grid, levels=levels)

    data = presets[0][0](*grid, **presets[0][1])

    for preset, kwargs, op in presets[1:]:
        data = op(data, preset(*grid, **kwargs))

    for timepattern, kwargs in timepatterns:
        data = timepattern(data, **kwargs)

    writeIEEEbin(args.out + ".bin", data, precision=args.precision)

    if args.npy:
        np.save(args.out + ".npy", data)

    if args.fill:
        fill_datafield(args.fill, data)

    if not args.no_plot:
        print("Generating figure.")
        figtitle = "Initial data of preset {} ({})".format(
            "+".join([p.name for p, k, o in presets]), args.out + ".bin"
        )
        with plt.style.context("ggplot"):
            plt.matplotlib.rcParams["axes.grid"] = False
            make_figure(
                data, figFile=args.out + ".png", title=figtitle, show=args.show_fig
            )


def make_figure(data, figFile=None, title=None, show=False):
    """
    Generate a figure showing slices of data.

    6 subplots with slices at 0 and middle for all directions.

    If figname is not None, saves to a file.
    If show is True, shows the figure (plt.show).
    """
    vmax = data.max()
    vmin = data.min()

    # Smart choice between divergent and sequential colormaps
    # If data is spread around 0, takes a divergent colormap.
    if (vmin * vmax != 0) and (4 > -vmax / vmin > 0.25):
        cmap = plt.cm.RdBu
        vmax = max(vmax, -vmin)
        vmin = -vmax
    else:
        cmap = plt.cm.viridis

    if data.ndim == 3:  # For 3D data
        fig, (
            (axZ, axZX, axZY, axN1, axYX, axC),
            (axN2, axX, axY, axN3, axN4, axN5),
        ) = plt.subplots(
            2,
            6,
            figsize=(16, 6),
            gridspec_kw={
                "width_ratios": [1, 4, 4, 1.5, 4, 0.25],
                "height_ratios": [4, 1],
            },
        )

        axZ.plot(data.mean(axis=(0, 1)), np.arange(data.shape[2]))
        axY.plot(data.mean(axis=(0, 2)))
        axX.plot(data.mean(axis=(1, 2)))

        ixyz = np.array(data.shape) // 2
        im = axZX.imshow(
            data[:, ixyz[1], :].T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto"
        )
        axZY.imshow(
            data[ixyz[0], ...].T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto"
        )
        axYX.imshow(
            data[:, :, ixyz[2]].T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto"
        )

        axX.set_xlim(axZX.get_xlim())
        axY.set_xlim(axZY.get_xlim())
        axZ.set_ylim(axZX.get_ylim())

        fig.colorbar(im, cax=axC)
        axN1.set_visible(False)
        axN2.set_visible(False)
        axN3.set_visible(False)
        axN4.set_visible(False)
        axN5.set_visible(False)

        axX.set_xlabel("X")
        axY.set_xlabel("Y")
        axY.yaxis.set_ticks_position("right")

        axZ.xaxis.set_ticks_position("top")
        axZ.set_ylabel("Z")

        axYX.set_ylabel("Y")
        axYX.set_xlabel("X")

        axZX.set_xticks([])
        axZX.set_yticks([])
        axZY.yaxis.set_ticks_position("right")
        axZY.set_ylabel("Z")
        axZY.yaxis.set_label_position("right")
        axZY.set_xticks([])
        axZY.spines["left"].set_color("k")

        for i, ax, lbl in zip(ixyz, [axZY, axZX, axYX], ["X", "Y", "Z"]):
            ax.set_title("{lbl} = {i:d}".format(i=i, lbl=lbl))

        if title is not None:
            fig.suptitle(title, wrap=True)

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout(h_pad=0, w_pad=0)

        t = axX.yaxis.get_offset_text()
        if t:
            axX.set_ylabel(t.get_text())
            t.set_visible(False)

        t = axY.yaxis.get_offset_text()
        if t:
            axY.set_ylabel(t.get_text())
            axY.yaxis.set_label_position("right")
            t.set_visible(False)

        t = axZ.xaxis.get_offset_text()
        if t:
            axZ.set_xlabel(t.get_text())
            axZ.xaxis.set_label_position("top")
            t.set_visible(False)

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.subplots_adjust(top=0.9, wspace=0, hspace=0)
    else:  # For 2D data, one simple plot
        fig, ((axY, ax, axC), (axN1, axX, axN2)) = plt.subplots(
            2,
            3,
            figsize=(7, 6),
            gridspec_kw={"width_ratios": [1, 4, 0.2], "height_ratios": [4, 1]},
        )

        im = ax.imshow(data.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axY.plot(data.mean(axis=0), np.arange(data.shape[1]))
        axY.set_ylim(ax.get_ylim())
        axX.plot(data.mean(axis=1))
        axX.set_xlim(ax.get_xlim())

        fig.colorbar(im, cax=axC)
        axN1.set_visible(False)
        axN2.set_visible(False)

        axX.set_xlabel("x")
        axY.xaxis.set_ticks_position("top")
        axY.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

        if title is not None:
            ax.set_title(title, wrap=True)

        ax.set_aspect("auto")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout(h_pad=0, w_pad=0)

        t = axX.yaxis.get_offset_text()
        if t:
            axX.set_ylabel(str(t.get_text()))
            t.set_visible(False)

        t = axY.xaxis.get_offset_text()
        if t:
            axY.set_xlabel(str(t.get_text()))
            axY.xaxis.set_label_position("top")
            t.set_visible(False)

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.subplots_adjust(wspace=0, hspace=0)

    if figFile is not None:
        fig.savefig(figFile)

    if show:
        plt.show()


if __name__ == "__main__":
    main()
