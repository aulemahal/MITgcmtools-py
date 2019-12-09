# -*- coding: utf-8 -*-
"""
Quantites calculator

Functions that compute additional quantities into netCDF datasets.
Uses xarray's Dataset an xmitgcm to read from a MITgcm's run directory.
Uses xgcm and xrft for the computations. All registered function should have the following signature:

>>> def funcname(dataset, grid, **kwargs):

The CLI function diagnose() will parse the "data" file and pass it entirely to the functions. Kwargs can also be given in the cli call.
Finally, all previously computed variables will be passed.

If called with -n X, created a dask Client with X workers.

Constants:
    DIAGNOSTICS : List of dicts describing the quantities computation functions available through the diagnose function / cli interface.

Base functions / object defined:
    add_diag       : Decorator to add functions to the DIAGNOSTICS constant
    list_diagnostics : List all available diagnostics and their properties.
    diagnose       : Cli interface entry-point.

Typical function:

@add_diag(NAME, need_raw=[list of fields needed in the input dataset], need_add=[list of fields to be pre-computed])
function(input dataset (xarray Dataset), xgcm Grid of the input, **kwargs):
    var = stuff
    var.attrs.update(name= ,long_name=, standard_name=, units=)
    return var
"""
import argparse
import numpy as np
import numba as nb
import xarray as xr
import inspect as ins
from xgcm import Grid
from pathlib import Path
from warnings import warn
from datetime import datetime as dt
from scipy.signal import windows, savgol_filter
from scipy.interpolate import interp1d
from .io import open_runfolder
from .jmd95 import densjmd95
from .common import COLORS, parse_kwarg

try:
    import gsw

    GSWVARS = {"SA": "SALT", "CT": "THETA", "p": "PHrefC"}
except ImportError:
    warn("Diagnostics GSW unavailable since package gsw is missing.")
    GSWVARS = None
try:
    from dask.diagnostics import ProgressBar
except ImportError:

    class ProgressBar:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


DIAGNOSTICS = {}
MITgcm_COORD_NAMES = {
    "X": ["XC", "XG"],
    "Y": ["YC", "YG"],
    "Z": ["Z", "Zl", "Zp1", "Zu"],
}


def add_diag(diagname, needs=None, dataset=False, order=None, parent=None):
    """Decorator to add a function to the DIAGNOSTICS constant list with some kwargs.
       `needs can be a string (or list of) referring to a variable that needs to be in the dataset or available in DIAGNOSTICS
        both can also be "$XXX", where XXX is a keyword arg of the diag func, than needs will be filled on call time.

        If Dataset is True, the diag must be a function that acts on the full dataset and returns a new dataset
        If it is False, the diag must be a function that returns one or more variable.
            Variables will be assigned with the name as:  variables.attrs.get('name', diagname) [So name *must* be set for multiple diagnostics]
        Order is ignored if Dataset is False.

        All functions must have the signature : dataset, grid, **kwargs
    """
    needs = [needs] if isinstance(needs, str) and not needs.startswith("$") else needs

    def _add_diag(func):
        kwarglist = [
            k
            for k, v in ins.signature(func).parameters.items()
            if v.default is not ins._empty
        ] + (
            []
            if parent is None
            else [
                k
                for k, v in ins.signature(parent).parameters.items()
                if v.default is ins._empty
            ]
        )
        DIAGNOSTICS[diagname] = {
            "name": diagname,
            "func": func,
            "parent": parent,
            "needs": needs,
            "kwarglist": kwarglist,
            "dataset": dataset,
            "order": order,
        }
        return func

    return _add_diag


def _find_coord(var, firstletter="Z"):
    for coord in var.dims:
        if coord.startswith(firstletter):
            return coord
    return None


@add_diag("vort3", needs=["UVEL", "VVEL"])
def vort3(d, g=None, **kwargs):
    """Vorticity ζ = dudv - dvdx"""
    dxC = d.get("dxC", d.XC[1] - d.XC[0])
    dyC = d.get("dyC", d.YC[1] - d.YC[0])
    rAz = d.get("rAz", dxC * dyC)
    zeta = (
        -g.diff(
            d.UVEL * dxC,
            "Y",
            boundary="" if g.axes["Y"]._periodic else "fill",
            fill_value=np.nan,
        )
        + g.diff(  # The long if else is manage cases where I take only
            d.VVEL * dyC,
            "X",
            boundary="" if g.axes["X"]._periodic else "fill",
            fill_value=np.nan,
        )
    ) / rAz  # a subset of the domain, thus with incomplete
    zeta.attrs.update(
        name="vort3",
        long_name="Z-component of vorticity",
        standard_name="momVort3",
        units="s^-1",
    )  # coords that are not periodic anymore
    return zeta


@add_diag("vort2", needs=["UVEL", "WVEL"])
def vort2(d, g, **kwargs):
    """Vorticity ω_2 = dudz - dwdx"""
    if "drC" in d:
        drC = xr.DataArray(
            d.drC[:-1].values, dims=("Zl",), coords={"Zl": d.Zl}, name="drC"
        )
    else:
        drC = abs(d.Zl[1] - d.Zl[0])
    dxC = d.get("dxC", d.XC[1] - d.XC[0])
    rAw = d.get("rAw", dxC * drC)
    zeta = (
        g.diff(d.UVEL * dxC, "Z", boundary="fill", fill_value=np.nan)
        * np.sign(d.Zl[1] - d.Zl[0])
        - g.diff(
            d.WVEL * abs(drC),
            "X",
            boundary="" if g.axes["X"]._periodic else "fill",
            fill_value=np.nan,
        )
    ) / rAw
    zeta.attrs.update(
        name="vort2",
        long_name="Y-component of vorticity",
        standard_name="momVort2",
        units="s^-1",
    )
    return zeta


@add_diag("vort1", needs=["VVEL", "WVEL"])
def vort1(d, g, **kwargs):
    """Vorticity ω_1 = dvdz - dwdy"""
    if "drC" in d:
        drC = xr.DataArray(
            d.drC[:-1].values, dims=("Zl",), coords={"Zl": d.Zl}, name="drC"
        )
    else:
        drC = abs(d.Zl[1] - d.Zl[0])
    dyC = d.get("dyC", d.YC[1] - d.YC[0])
    rAs = d.get("rAs", dyC * drC)
    zeta = (
        g.diff(
            d.WVEL * abs(drC),
            "Y",
            boundary="" if g.axes["Y"]._periodic else "fill",
            fill_value=np.nan,
        )
        - g.diff(d.VVEL * dyC, "Z", boundary="fill", fill_value=np.nan)
        * np.sign(d.Zl[1] - d.Zl[0])
    ) / rAs
    zeta.attrs.update(
        name="vort1",
        long_name="X-component of vorticity",
        standard_name="momVort1",
        units="s^-1",
    )
    return zeta


@nb.njit
def _jacobi_NN(G, F, R, d=1):
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            R[i, j] = (
                G[i - 1, j]
                + G[i + 1, j]
                + G[i, j - 1]
                + G[i, j + 1]
                - 4 * G[i, j]
                - d ** 2 * F[i, j]
            )


@nb.njit
def _jacobi_NP(G, F, R, d=1):
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            R[i, j] = (
                G[i - 1, j]
                + G[i + 1, j]
                + G[i, j - 1]
                + G[i, j + 1]
                - 4 * G[i, j]
                - d ** 2 * F[i, j]
            )
        R[i, 0] = (
            G[i - 1, 0]
            + G[i + 1, 0]
            + G[i, -1]
            + G[i, 1]
            - 4 * G[i, 0]
            - d ** 2 * F[i, 0]
        )
        R[i, -1] = (
            G[i - 1, -1]
            + G[i + 1, -1]
            + G[i, -2]
            + G[i, 0]
            - 4 * G[i, -1]
            - d ** 2 * F[i, -1]
        )


@nb.njit
def _jacobi_PP(G, F, R, d=1):
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            R[i, j] = (
                G[i - 1, j]
                + G[i + 1, j]
                + G[i, j - 1]
                + G[i, j + 1]
                - 4 * G[i, j]
                - d ** 2 * F[i, j]
            )
    for j in range(1, R.shape[1] - 1):
        R[0, j] = (
            G[-1, j]
            + G[1, j]
            + G[0, j - 1]
            + G[0, j + 1]
            - 4 * G[0, j]
            - d ** 2 * F[0, j]
        )
        R[-1, j] = (
            G[-2, j]
            + G[0, j]
            + G[-1, j - 1]
            + G[-1, j + 1]
            - 4 * G[-1, j]
            - d ** 2 * F[-1, j]
        )
    for i in range(1, R.shape[0] - 1):
        R[i, 0] = (
            G[i - 1, 0]
            + G[i + 1, 0]
            + G[i, -1]
            + G[i, 1]
            - 4 * G[i, 0]
            - d ** 2 * F[i, 0]
        )
        R[i, -1] = (
            G[i - 1, -1]
            + G[i + 1, -1]
            + G[i, -2]
            + G[i, 0]
            - 4 * G[i, -1]
            - d ** 2 * F[i, -1]
        )


def _psi(vort, face="PP", guess=None, verbose=False, rtol=1e-4, itmax=12000, omega=1):
    """
    Parameters
    ----------
    vort   : Vorticity ND array. If not 2D, iterates on all dimensions but the two last ones.
    face   : one of 'PP', 'NP', 'NN'.
    method : Method to pass to the Newton Krylov function.
    verbose : Verbosity level of the solver.
    """
    if vort.ndim > 2:
        result = np.empty_like(vort)
        for idxs in np.ndindex(vort.shape[:-2]):
            result[idxs] = _psi(
                vort[idxs],
                face=face,
                guess=guess,
                verbose=verbose,
                rtol=rtol,
                itmax=itmax,
                omega=omega,
            )
        return result

    if np.any(np.isnan(vort)):
        warn("NaN value(s) encoutered in vorticity, whole slice of Psi set to NaN.")
        return vort * np.nan

    guess = guess or np.zeros_like(vort)
    residuals = np.zeros_like(vort)
    vort_max = np.max(vort)
    jacobi_fun = {"NN": _jacobi_NN, "NP": _jacobi_NP, "PP": _jacobi_PP}[face]

    for it in range(itmax):
        jacobi_fun(guess, vort, residuals)
        guess = guess + (omega / 4) * residuals
        if np.max(residuals) / vort_max < rtol:
            if verbose:
                print("Convergence obtained in {} iterations.".format(it))
            return guess

    mess = "No convergence obtained after {} iterations. max(R)/max(w) = {} > {}".format(
        it, np.max(residuals) / vort_max, rtol
    )
    if verbose:
        print(mess)
    else:
        warn(mess)
    return guess


def Psi(d, axis="Z", **kwargs):
    """Computation of the streamfunction DataArray through the Poisson Equation definition:
    ∇²Ψ = ω

    Parameters
    ----------
    axes : Axes to be used. Others will be stacked.
    """
    if axis == "Z":
        face, ax = "PP", "z"
        cdims = ("YG", "XG")
        vort = d.vort3
    elif axis == "Y":
        face, ax = "NP", "y"
        cdims = ("Zl", "XG")
        vort = d.vort2
    else:  # axis == 'X'
        face, ax = "NN", "x"
        cdims = ("Zl", "YG")
        vort = d.vort1

    if "Zl" in vort.dims and vort.isel(Zl=0).isnull().all():
        vort = vort.isel(Zl=slice(1, None))

    psi = xr.apply_ufunc(
        _psi,
        vort,
        dask="parallelized",
        input_core_dims=[cdims],
        output_core_dims=[cdims],
        kwargs=dict(face=face, **kwargs),
        output_dtypes=[float],
    )
    psi.attrs.update(
        name="psi",
        standard_name=r"$\Psi_" + ax + "$",
        long_name="Streamfunction in {}".format(ax),
        units="m²/s",
    )
    return psi


@add_diag("PsiZ", needs=["vort3"], parent=_psi)
def PsiZ(d, g, **kwargs):
    """Computation of the streamfunction on z through the Poisson Equation definition:
    ∇²Ψ = ζ
    """
    psi = Psi(d, axis="Z", **kwargs)
    psi.attrs.update(name="Psi_Z")
    return psi


@add_diag("PsiY", needs=["vort2"], parent=_psi)
def PsiY(d, g, **kwargs):
    """Computation of the streamfunction on z through the Poisson Equation definition:
    ∇²Ψ = ω_y
    """
    psi = Psi(d, axis="Y", **kwargs)
    psi.attrs.update(name="Psi_Y")
    return psi


@add_diag("PsiX", needs=["vort1"], parent=_psi)
def PsiX(d, g, **kwargs):
    """Computation of the streamfunction on z through the Poisson Equation definition:
    ∇²Ψ = ω_1
    """
    psi = Psi(d, axis="X", **kwargs)
    psi.attrs.update(name="Psi_X")
    return psi


@add_diag("Density", needs=["SALT", "THETA", "PHrefC"])
def density(d, g, potential=False, **kwargs):
    """Density as computed in the MITgcm with method JMD95. Potential density added."""
    density = densjmd95(d.SALT, d.THETA, d.PHrefC * (not potential))

    if potential:
        density.attrs.update(
            name="Pot. Density",
            long_name="Pot. Density (JMD95)",
            standard_name=r"\sigma",
            units="kg/m³",
        )
    else:
        density.attrs.update(
            name="Density",
            long_name="Density (JMD95)",
            standard_name=r"\rho",
            units="kg/m³",
        )
    return density


def wt_turb_tracs(data, g, dtdz=None, N=7, win=None):
    """Compute wT for all PTracers in data"""
    d = data.chunk({"time": _best_chunk_size(data.time.size, np.ceil(N / 2))})

    win = win or windows.gaussian(N, 1, sym=True)
    weight = xr.DataArray(win, dims=["window"])

    tracs = sorted([trac for trac in d.variables if trac.startswith("TRAC")])
    if dtdz is None:
        dtdz = [
            d[trac].isel(XC=0, YC=0, time=1, Z=slice(-3, -1)).diff("Z")[0]
            for trac in tracs
        ]

    W = g.interp(d.WVEL, "Z", to="center", boundary="fill").isel(Z=slice(0, -1))
    Wb = W.rolling(time=N, center=True).construct("window").dot(weight)
    PTbs = {
        trac: d[trac]
        .isel(Z=slice(0, -1))
        .rolling(time=N, center=True)
        .construct("window")
        .dot(weight)
        for trac in tracs
    }
    wPTs = [
        (Wb * PTbs[trac])
        - (W * d[trac].isel(Z=slice(0, -1)))
        .rolling(time=N, center=True)
        .construct("window")
        .dot(weight)
        for trac in tracs
    ]
    wPT = xr.concat(wPTs, dim="dTdz")
    wPT = wPT.assign_coords(dTdz=dtdz)
    wPT.attrs.update(
        name="wT",
        standard_name="w'T'",
        long_name="Vertical Turbulent Heat Flux",
        units="K m s-1",
    )
    wPT.dTdz.attrs.update(
        name="dTdz",
        standard_name="dT/dz",
        long_name="Vertical Temperature Gradient under the ML",
        units="K m-1",
    )

    PTB = xr.concat([PTbs[trac] for trac in tracs], dim="dTdz")
    PTB.assign_coords(dTdz=dtdz)
    PTB.attrs.update(
        name="TRACSB",
        standard_name=r"$\bar{T}$",
        long_name="Averaged Linear Temperatures",
        units="K",
    )
    PTB.dTdz.attrs.update(
        name="dTdz",
        standard_name="dT/dz",
        long_name="Vertical Temperature Gradient under the ML",
        units="K m-1",
    )

    Wb.attrs.update(
        name="WVELB",
        standard_name=r"$\bar{w}$",
        long_name="Averaged Vertical Velocity",
        units="m s-1",
    )

    return wPT, PTB, Wb


def wt_turb(d, g, var="THETA", N=7, win=None):
    """Compute the vertical turbulent flux of Temperature or any other variable at tracer gridpoints.

    Uses <w'T'> = <w><T> - <wT>

    w is WVEL and is interpolated from Zl to Z. Last Z is discarded from both w and T.

    PARAMETERS
    ----------
    d, g : The Dataset and its Grid
    var  : str, name of the variable. Must have coords XC, YC and Z
    N    : int, number of points for the running mean.
    win  : seq of float, the weighting window to use, default : scipy.windows.gaussian with sigma=1.

    RETURNS
    -------
    <w'T'>, <T>, <w> : xarray Datasets
    """
    data = d.chunk({"time": _best_chunk_size(d.time.size, np.ceil(N / 2))})

    win = win or windows.gaussian(N, 1, sym=True)
    weight = xr.DataArray(win, dims=["window"])

    W = g.interp(data.WVEL, "Z", to="center", boundary="fill").isel(Z=slice(0, -1))
    Wb = W.rolling(time=N, center=True).construct("window").dot(weight)
    Tb = (
        data[var]
        .isel(Z=slice(0, -1))
        .rolling(time=N, center=True)
        .construct("window")
        .dot(weight)
    )
    wT = (Wb * Tb) - (W * data[var].isel(Z=slice(0, -1))).rolling(
        time=N, center=True
    ).construct("window").dot(weight)

    if var == "THETA":
        wT.attrs.update(
            name="wT",
            standard_name="w'T'",
            long_name="Vertical Turbulent Heat Flux",
            units="K m s-1",
        )
    else:
        wT.attrs.update(
            name="w·" + var,
            standard_name="Turbulent vertical fluxes of "
            + data[var].attrs.get("standard_name", var),
            units=data[var].attrs.get("units", "-") + " m s-1",
        )
    Tb.attrs.update(
        name=var + "B",
        standard_name="Averaged " + data[var].attrs.get("standard_name", var),
        long_name="Averaged " + data[var].attrs.get("long_name", var),
        units=data[var].attrs.get("units", ""),
    )
    Wb.attrs.update(
        name="WVELB",
        standard_name=r"$\bar{w}$",
        long_name="Averaged Vertical Velocity",
        units="m s-1",
    )

    return wT, Tb, Wb


def _best_chunk_size(totsize, minsize):
    minsize = int(minsize)
    for size in range(minsize, 2 * minsize):
        if (totsize % size == 0) or (totsize % size >= minsize):
            return size
    return 2 * minsize


@add_diag("wT", needs=["WVELTH", "WVEL", "THETA"])
def wT(d, g, cp=4000, rho=1000):
    """Turbulent vertical heat fluxes : <wT> - <w><T>"""
    wT = (
        cp
        * rho
        * (
            d.WVELTH
            - d.WVEL
            * g.interp(d.THETA, "Z", to="left", boundary="fill", fill_value=np.nan)
        )
    )
    wT.attrs.update(
        name="wT",
        standard_name="Vertical_heat_fluxes",
        long_name="Vertical turbulent heat fluxes",
        units="Wm-2",
    )
    return wT


@add_diag("deltamld", needs="$var")
def zchange(d, g, var=None, init="Tini", window=(-15, -25), zrange=(0, 20, 0.25)):
    """Compute the vertical displacement of a profile by find the dz of maximal rms correlation."""
    DZs = np.arange(*zrange)

    def _zchange(va, ini):
        if (isinstance(ini, str) and ":" in ini) or (
            np.isscalar(ini) and np.isreal(ini)
        ):
            V0 = d[va].sel(time=ini)
        elif isinstance(ini, str) and ini in d:
            V0 = d[ini]
        else:
            V0 = ini

        Z = _find_coord(d[va], "Z")
        z0 = V0[Z].sel({Z: slice(*window)})
        Vs = xr.concat(
            [V0.interp({Z: z0 - dz}) for dz in DZs],
            dim=xr.DataArray(DZs, dims=("dz",), name="dz"),
        )

        return zrange[0] + np.sqrt(((d[va] - Vs) ** 2).mean(Z)).argmin("dz") * zrange[2]

    if isinstance(var, list):
        if (isinstance(init, str) and ":" in init) or (
            np.isscalar(init) and np.isreal(init)
        ):
            init = [init] * len(var)
        zchange = xr.concat(
            [_zchange(va, ini) for va, ini in zip(var, init)], dim="vars"
        ).mean("vars")
        zchange.attrs.update(
            name="z_change",
            units=d["Z"].attrs.get("units"),
            long_name="Z-change from {}".format(",".join(var)),
        )
        return zchange
    zchange = _zchange(var, init)
    zchange.attrs.update(
        name="{var}_dMLD".format(var=var),
        units=d["Z"].attrs.get("units"),
        long_name="ΔMLD from {var}".format(var=var),
    )
    return zchange


# @add_diag('SPEC')
# def spectra(dataset, outf, cell_size, var=None, Nk=None, perlevel=True, mode='psd'):
#     """
#     Compute the radial spectrum of a given variable.

#     var gives the variable to use. "VEL" means the complete velocity spectrum.
#     Nk is a int, the number of bins the wavenumber space. (default: half of the smallest number of points in a direction (Ex: 150 for a 300x400x500 array))
#         By default, the bins range from 0 to the nyquist freq (pi / dx).
#         If Nk < 0, they will range from 0 to the largest possible bin in Fourier Space.
#     perlevel, if true (default), computes a 2D spectrum per vertical level instead of a 3D spectrum.

#     Each spectrum is a 1D array E(k) of the energy density at a given radial wavenumber K so that KEtot = ∫ E(k) dk
#     It is in Joules/m for VEL.

#     Adds {var}_SpecXD to the ncout dataset. Where X is 2 with perlevel and 3 without.
#     Also adds the K dimension and variable if missing. CAREFUL! use the same Nk for all spectra in one dataset!
#     """
#     # So the rest stays general, data is always a list
#     ndim = 2 if perlevel else 3
#     if var == 'VEL':
#         vvar = 'UVEL,VVEL' if perlevel else 'UVEL,VVEL,WVEL'
#     else:
#         vvar = var
#     data = [dataset[vv] for vv in vvar.split(',')]

#     data_shape = [data[0].shape[-3], dataset.dimensions['Y'].size, dataset.dimensions['X'].size]
#     slc = [slice(0, N) for N in data_shape]
#     cell_size = list(reversed(cell_size[:ndim]))

#     kappas = mitspec.get_kappas(data_shape[-ndim:], cell_size)
#     k_bins = mitspec.get_k_bins(kappas=kappas, cell_size=cell_size, Nk=Nk, data_shape=data_shape[-ndim:])

#     # As usual, create the variable. Here we need to create K also, so we don't use GetNewVariable.
#     var_name = var.replace(',', '_') + '_' + ('Cospec' if mode == 'cspc' else 'Spec') + ('2D' if perlevel else '3D')
#     if MPI_RANK == 0:
#         with nc.Dataset(outf, 'a') as outd:
#             if 'K' not in outd.variables:
#                 outd.createDimension('K', size=len(k_bins) - 1)
#                 K_var = outd.createVariable('K', k_bins.dtype, dimensions='K')
#                 K_var.setncattr('units', '1/m')
#                 K_var.setncattr('description', 'Radial wavenumber')
#                 K_var[:] = (k_bins[:-1] + k_bins[1:]) / 2

#             spec_var = outd.createVariable(var_name, np.float, dimensions=('T', data[0].dimensions[1], 'K') if perlevel else ('T', 'K'))
#             spec_var.setncattr('units', 'J')
#             spec_var.setncattr('description', ('2D' if perlevel else '3D') + ' ' + ('co' if mode == 'cspc' else '') + 'spectrum of ' + var.replace(',', ' and '))
#     if MPI_SIZE > 1:
#         MPI.COMM_WORLD.Barrier()

#     with nc.Dataset(outf, 'a', parallel=(MPI_SIZE > 1)) as outd:
#         spec_var = outd[var_name]

#         # Pre compute all bin masks to save time. kmask is an array with an int indices of the corresponding lower edge kappa in k_bins.
#         k_mask = mitspec.get_k_mask(kappas, k_bins)

#         for it in mpi_range(data[0].shape[0]):
#             spec_var[it, ...] = mitspec.spectrum([dat[[it] + slc] for dat in data], cell_size, ndim=ndim,
#                                                  k_bins=k_bins, kappas=kappas, k_mask=k_mask, mode=mode)


@nb.njit
def _mld(col, Z, thresh):
    for iz in range(1, len(col) - 1):
        if col[iz] > thresh and col[iz + 1] > thresh:
            return Z[iz] + (Z[iz - 1] - Z[iz]) * (col[iz] - thresh) / (
                col[iz] - col[iz - 1]
            )
    return np.nan


def _mld_chunk(chunk, Z=None, smoothed=False, threshold=None):
    if chunk.ndim > 1:
        result = np.empty(chunk.shape[:-1])
        for idxs in np.ndindex(chunk.shape[:-1]):
            # MITgcm saves in > byte order, which numba doesn't support
            col = abs(chunk[idxs]).astype(float)
            if smoothed:
                col = savgol_filter(col, 5, 4)
            result[idxs] = _mld(col, Z, threshold)
        return result
    if smoothed:
        col = savgol_filter(chunk, 5, 4)
    return _mld(col, Z, threshold)


@add_diag("mld", needs="$var")
def mixed_layer_depth(
    d, g, var=None, threshold=0.02, smoothed=False, dask="parallelized", **roll_kwargs
):
    """
    Compute the Mixed Layer Depth as the first depth that some variable's vertical derivative exceeds a threshold.
    The first layer is exluded from the computation.

    PARAMETERS
    ----------
    threshold : absolute value of threshold of the vertical derivative of "var" higher of which it is considered a ML
    var       : variable to look at, default is Density (from add_diag density)
    smooted   : If True, applies a 5-levels 4th order Savitzky-Golay filter to each columns.
    roll_kwargs : Mapping of dim : window size for rolling averages done on the vertical derivative field before.
    """
    Dz = g.diff(d[var], "Z", boundary="fill", fill_value=np.nan)
    for dim, win in roll_kwargs.items():
        Dz = Dz.rolling(dim={dim: win}, center=True, min_periods=1).mean()
    Z = _find_coord(Dz, "Z")
    if dask == "parallelized":
        Dz = Dz.chunk({Z: -1})
    # MITgcm saves in > byte order, which numba doesn't support
    mld = xr.apply_ufunc(
        _mld_chunk,
        Dz,
        input_core_dims=[(Z,)],
        dask=dask,
        output_dtypes=[float],
        kwargs=dict(
            Z=d[Z].values.astype(float), smoothed=smoothed, threshold=threshold
        ),
    )
    mld.attrs.update(
        name="MLD",
        long_name="Mixed Layer Depth",
        description=f"Computed from {var} with dF/dz >= {threshold}",
        units=d[Z].attrs.get("units", "m"),
    )
    return mld


@add_diag("GSW", needs="$varargs")
def from_gsw(d, g, func=None, varargs=None, dask="parallelized", **kwargs):
    """Wrapper fo any diagnostic available in gsw."""
    if GSWVARS is None:
        raise NotImplementedError(
            "Package GSW is missing, thus from_gsw cannot be implemented."
        )
    if isinstance(func, str):
        func = getattr(gsw, func)

    if varargs is None:
        varargs = map(
            GSWVARS.get,
            [
                k
                for k, v in ins.signature(func).parameters.items()
                if v.default is ins._empty
            ],
        )

    N = -1
    for line in map(str.strip, func.__doc__.split("\n")):
        if "Returns" in line:
            N = 3
        elif N == 1:
            name, units = line.split(":")
            name = name.strip()
            units = units.split(",")[-1].strip()
        elif N == 0:
            long_name = line
            break
        N -= 1

    data = xr.apply_ufunc(
        func,
        *[d[var] for var in varargs],
        kwargs=kwargs,
        dask=dask,
        output_dtypes=[float],
    )
    data.attrs.update({"long_name": long_name, "units": units, "name": name})
    return data


def _interp1DAt(var, coord, X=None):
    if var.ndim > 1:
        result = np.empty(var.shape[:-1])
        for idxs in np.ndindex(var.shape[:-1]):
            result[idxs] = _interp1DAt(var[idxs], coord[idxs], X=X)
        return result
    return interp1d(X, var)(coord)


@add_diag("InterpAt", needs="$varargs")
def interp_at(d, g, varargs=None, dim=None, dask="parallelized"):
    """
    Interpolates a variable to another.
    Example : varargs = [THETA, mld] : THETA(t, z, y, x) is interpolated with Z=mld(t, y, x)
    """
    var, coordvar = varargs
    dim = (
        dim if dim is not None else set(d[var].dims).difference(d[coordvar].dims).pop()
    )
    X = d[dim].values
    data = xr.apply_ufunc(
        _interp1DAt,
        d[var],
        d[coordvar],
        input_core_dims=[[dim], []],
        dask=dask,
        output_dtypes=[float],
        kwargs={"X": X},
        keep_attrs=True,
    )
    data.attrs.update(
        long_name=d[var].attrs.get("long_name", var)
        + " interpolated to {} along {}".format(coordvar, dim),
        name="{}_{}_{}".format(var, dim, coordvar),
    )
    return data


@add_diag("Roll_dataset", dataset=True)
def roll_dataset(d, g, roll_dim="X", U=-0.05):
    """Rolls a whole datasets, rolls on roll_dim of amount U*t for each t."""
    for varname in d.variables.keys():
        X = _find_coord(d[varname], roll_dim)
        if (X is not None) and ("time" in d[varname].dims):
            time_sec = d.time.values.astype("timedelta64[s]")
            for it, tim in enumerate(time_sec):
                d[varname][it, :] = (
                    d[varname]
                    .isel(time=it)
                    .roll(shifts={X: (tim * U).astype(int)}, roll_coords=False)
                )
            d[varname].attrs.update(
                description=d[varname].attrs.get("description", "")
                + "Rolled on {} by {} m/s".format(X, U)
            )
    return d


def list_diagnostics():
    """List all diagnostics, giving info about the needed fields, arguments and docstrings."""
    outStr = (
        COLORS.LIGHTBLUE + COLORS.BOLD + "Available diagnostics:" + COLORS.ENDC + "\n"
    )
    for diag in DIAGNOSTICS.values():
        outStr += COLORS.YELLOW + diag["name"] + COLORS.ENDC + "\n"
        outStr += (
            COLORS.BLUE
            + "\tDiagnostics needed (either raw or computed before): "
            + COLORS.ENDC
        )
        if isinstance(diag["needs"], str):
            outStr += "Given by argument : " + diag["needs"][1:]
        elif diag["needs"] is None:
            outStr += "None"
        else:
            outStr += ", ".join(diag["needs"])
        outStr += "\n"
        outStr += (
            COLORS.BLUE
            + "\tExtra arguments: "
            + COLORS.ENDC
            + ", ".join(diag["kwarglist"])
            + "\n"
        )
        outStr += (
            COLORS.PURPLE
            + "\tDoc-string:"
            + COLORS.ENDC
            + "\n\t".join(
                diag["func"].__doc__.split("\n")
                + ([] if diag["parent"] is None else diag["parent"].__doc__.split("\n"))
            )
        )
        outStr += "\n"
    return outStr


def _get_output_specs(outputs, input_file):
    """Parse the output specs given to diagnose."""
    specs = []
    for output in outputs:
        means = []
        coord_slices = {}
        idx_slices = {}
        *params, outfile = output.split(",")
        for spec in params:
            coord = "X" if "X" in spec else ("Y" if "Y" in spec else "Z")
            mean, slicespec = spec.split(coord)
            if ":" in slicespec:
                start, end, *step = slicespec.split(":")
                start = (
                    None
                    if not start
                    else (float(start) if "." in start else int(start))
                )
                end = None if not end else (float(end) if "." in end else int(start))
                step = None if len(step) == 0 else int(step[0])
                if isinstance(start, float) or isinstance(end, float):
                    coord_slices.update(
                        {
                            cname: slice(start, end, step)
                            for cname in MITgcm_COORD_NAMES[coord]
                        }
                    )
                else:
                    idx_slices.update(
                        {
                            cname: slice(start, end, step)
                            for cname in MITgcm_COORD_NAMES[coord]
                        }
                    )
            if mean == "m":
                means.extend(MITgcm_COORD_NAMES[coord])
        if outfile.startswith("_"):
            outfile = input_file.stem + outfile
        if not outfile.endswith(".nc"):
            outfile += ".nc"
        specs.append([coord_slices, idx_slices, means, outfile])
    return specs


def _get_index_of_diag(diaglist, diagname):
    """Helper function to get index"""
    for i, d in enumerate(diaglist):
        if d["name"] == diagname:
            return i
    return None


def _sort_diagnostics(diaglist, dataset, v=print):
    """Sort a list of diagnostics, inserting dependencies and solving cycles"""
    Npre = 0
    for N in range(100):  # A bit ugly, we limit to 100 iterations
        done = True
        for i in range(len(diaglist)):
            if diaglist[i]["dataset"]:  # Go through "dataset-diags" first
                if diaglist[i]["order"] == "pre":
                    done = False
                    diaglist.insert(Npre, diaglist.pop(i))
                    Npre += 1
                elif diaglist[i]["order"] == "post":
                    done = False
                    diaglist.append(diaglist.pop(i))
                diaglist[i]["order"] = "solved"  # So we don't loop again on them

            if isinstance(diaglist[i]["needs"], str) and diaglist[i][
                "needs"
            ].startswith("$"):
                # Dependencies given by a kwarg, silent if the kwarg wasn't given : diags should take care of this.
                done = False
                needed = diaglist[i]["kwargs"].get(diaglist[i]["needs"][1:])
                diaglist[i]["needs"] = [needed] if isinstance(needed, str) else needed

            if diaglist[i]["needs"] is not None:
                # There are some dependencies
                for needed in diaglist[i]["needs"]:
                    idxof = _get_index_of_diag(diaglist, needed)
                    if idxof is None:  # needed wasn't found in current diaglist
                        if (
                            needed in DIAGNOSTICS
                        ):  # Is it a Diag?, if yes, instert it before current diag
                            done = False
                            diaglist.insert(i, DIAGNOSTICS[needed].copy())
                            v(f'Adding {needed} before {diaglist[i + 1]["name"]}', 3)
                        elif needed in dataset:  # Is it a raw variable?
                            v(f"From dataset: {needed}", 3)
                        else:
                            raise ValueError(
                                f"Unable to find dependency {needed} in dataset or available diagnostics."
                            )
                    elif idxof > i:  # It is listed, but in the wrong order.
                        done = False
                        diaglist.insert(i, diaglist.pop(idxof))
                        v(f'Pushing {needed} before {diaglist[i + 1]["name"]}', 3)
        if done:  # A complete iteration was done without changes
            v(f"Done ordering diagnostics after {N} iterations")
            v("\n".join([f"{d['name']} (Needs: {d['needs']})" for d in diaglist]), 3)
            break
    else:  # Exceeded the max iterations
        v([f"{d['name']} (Needs: {d['needs']})" for d in diaglist], 3)
        raise RecursionError("Unable to solve the depencies properly.")
    return diaglist


def diagnose():
    """Entry-point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "CLI interface to compute complementary diagnostics on MITgcm data.\n"
            "Multiple diagnostics can be computed and saved in the same netCDF dataset.\n"
        )
    )
    parser.add_argument(
        "-l",
        "--list",
        help=("Lists all available diagnostics and their options."),
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "Input MITgcm run folder and prefix, or netCDF file. Defaults to current dir.\n"
            "Given as 'folder:prefix' or 'file.nc'. If no prefix is given, run_folder is called with merge_full=True."
        ),
        default=".",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            'Output netCDF file name. If it starts with "_", the input file/folder name is prepended\n'
            "Multiple output datasets are possible. Can be saved with slices or means as : [[m]Xi:j:n][Y...],file.nc"
            "If i or j have a ., they are treated as coordinates, if not as indexes (ints)"
        ),
        nargs="+",
    )
    parser.add_argument(
        "--dask",
        help="Control dask behavior for all diagnostics where it is possible (where `dask` kwarg exists)",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print stuff to the console, 1: General, 2+: Detailed",
        default=0,
        const=1,
        type=int,
        nargs="?",
    )
    parser.add_argument(
        "-d",
        "--diags",
        help='Diagnostics to compute with their keyword arguments listed as "arg=val"',
        nargs="+",
    )
    args = parser.parse_args()

    def v(mess, n=1):
        """A helper function to print messages depending on the verbosity level."""
        if args.verbose >= n:
            print(mess)

    if args.list:  # Print the diagnostics list and exit.
        print(list_diagnostics())
        return

    if args.output is None:
        parser.error("Must provide at least one output file")

    start = dt.now()

    # Read in each listed diags, copy the metadata and populate a kwargs dict
    diaglist = []
    for diagarg in args.diags or []:  # Avoiding error if no diag is given
        if "=" in diagarg:
            key, val = diagarg.split("=")
            diaglist[-1]["kwargs"][key] = parse_kwarg(val)
        else:
            diag = DIAGNOSTICS[diagarg]
            diaglist.append(diag.copy())
            diaglist[-1].update(kwargs={})
            if args.dask is not None and "dask" in diag["kwarglist"]:
                diaglist[-1]["kwargs"]["dask"] = args.dask

    if ".nc" in args.input:  # if input is a file
        dataset = xr.open_dataset(args.input)
        input_file = Path(args.input)
    else:  # if input is MITgcm run folder
        if ":" in args.input:  # folder + prefix
            folder, prefix = args.input.split(":")
        else:  # No prefix, assuming all diags
            folder = args.input
            prefix = "diag"
        input_file = Path(folder)
        GROUP_PRFS = ["all", "diag", "*"]
        dataset = open_runfolder(
            folder,
            prefixes=[prefix] if prefix not in GROUP_PRFS else prefix,
            merge_full=(prefix in GROUP_PRFS),
            verb=args.verbose > 1,
        )[prefix if prefix not in GROUP_PRFS else "full"]

    diaglist = _sort_diagnostics(diaglist, dataset, v=v)

    output_specs = _get_output_specs(args.output, input_file)

    outdataset = dataset.copy()

    grid = Grid(dataset, periodic=["X", "Y"])  # This is a MITgcm default

    for diag in diaglist:
        v(f"Computing / Assigning {diag['name']}")
        data = diag["func"](outdataset, grid, **diag.get("kwargs", {}))
        outdataset = outdataset.assign(
            **{data.attrs["name"]: data}
        )  # Diags can redefine a name attr

    for crd_slcs, idx_slcs, means, outfile in output_specs:
        # Select on coords, indexes and compute the means before saving to netCDF
        computation = (
            outdataset.pipe(lambda d: d if not crd_slcs else d.sel(**crd_slcs))
            .pipe(lambda d: d if not idx_slcs else d.isel(**idx_slcs))
            .pipe(lambda d: d if not means else d.mean(means))
            .to_netcdf(outfile, compute=False)
        )
        slicing_str = (
            f"(slicing on {','.join(list(crd_slcs.keys()) + list(idx_slcs.keys()))})"
        )
        averaging_str = f"(averaging on {','.join(means)})"
        v(
            f"Saving dataset to {outfile} {slicing_str if len(slicing_str) > 14 else ''} {averaging_str if means else ''}"
        )
        with ProgressBar():
            computation.compute()

    time = dt.utcnow() - start
    v(f"Everything done in {time.total_seconds():.0f} s", 2)
    dataset.close()


if __name__ == "__main__":
    diagnose()
