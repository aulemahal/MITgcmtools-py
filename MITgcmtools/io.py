# -*- coding: utf-8 -*-
"""
IO.py

Fonctions pour la lecture et l'écriture de expériences MITgcm
"""
import numpy as np
import xarray as xr
from pathlib import Path
from warnings import warn
from itertools import zip_longest
from collections import OrderedDict
from xmitgcm import open_mdsdataset
from .gendata import readIEEEbin


def _parse_namelist_val(val):
    """Parse a string and cast it in the appropriate python type."""
    if "," in val:  # It's a list, parse recursively
        return [_parse_namelist_val(subval.strip()) for subval in val.split(",")]
    elif val.startswith("'"):  # It's a string, remove quotes.
        return val[1:-1].strip()
    elif val in [".TRUE.", ".FALSE."]:
        return val == ".TRUE."

    try:
        if "*" in val:  # It's shorthand for a repeated value
            repeat, number = val.split("*")
            return [_parse_namelist_val(number)] * int(repeat)
        elif "." in val or "E" in val:  # It is a Real (float)
            return float(val)
        return int(val)
    except ValueError as err:
        raise err


def parse_namelist(file, parse=True, flat=False, silence_cast_errors=False):
    """Read a FOTRAN namelist file into a dictionary.

    PARAMETERS
    ----------
    file : str or Path
        Path to the namelist file to read.
    parse : bool
        Parse values to python types. If False, keep values as strings.
    flat : bool
        If True, flattens the output by merging all namelists of the file together.
    silence_cast_errors : bool
        If True, values that can't be casted are read as raw strings and a warning is issued
        If False, an error is raised.

    RETURNS
    -------
    data : dict
        Dictionary of each namelist as dictionaries (if flat is False)
                   of all keys and values in file (if flat is True)
                   All keys are lowercase.
    """

    data = {}
    current_namelist = ""
    raw_lines = []
    with open(file) as f:
        for line in f:
            # Remove comments
            line = line.split("#")[0].strip()
            if "=" in line or "&" in line:
                raw_lines.append(line)
            elif line:
                raw_lines[-1] += line

    for line in raw_lines:
        if line.startswith("&"):
            current_namelist = line.split("&")[1]
            if current_namelist:  # else : it's the end of a namelist.
                data[current_namelist] = {}
        else:
            field, value = map(str.strip, line[:-1].split("="))
            if parse:
                try:
                    value = _parse_namelist_val(value)
                except ValueError as err:
                    if silence_cast_errors:
                        warn(
                            "Unable to cast value {} at line {}".format(
                                value, raw_lines.index(line)
                            )
                        )
                    else:
                        raise err

            if "(" in field:  # Field is an array
                field, idxs = field[:-1].split("(")
                field = field.casefold()
                if field not in data[current_namelist]:
                    data[current_namelist][field] = []
                # For generality, we will assign a slice, so we cast in list
                value = value if isinstance(value, list) else [value]
                idxs = [
                    slice(int(idx.split(":")[0]) - 1, int(idx.split(":")[1]))
                    if ":" in idx
                    else slice(int(idx) - 1, int(idx))
                    for idx in idxs.split(",")
                ]

                datafield = data[current_namelist][field]
                # Array are 1D or 2D, if 2D we extend it to the good shape,
                # filling it with [] and pass the appropriate sublist.
                # Only works with slice assign (a:b) in first position.
                missing_spots = idxs[-1].stop - len(datafield)
                if missing_spots > 0:
                    datafield.extend([] for i in range(missing_spots))
                if len(idxs) == 2:
                    datafield = datafield[idxs[1].start]
                datafield[idxs[0]] = value
            else:
                data[current_namelist][field.casefold()] = value

    if flat:
        for namelist in list(data.keys()):
            data.update(data.pop(namelist))

    return data


def print_namelist(data, f):
    """Print a namelist dictionary to file.
    
    Parameters
    ----------
    data : dict
        Namelist as read by parse_namelist with parse=False
    f : file-like obj
        Obj exposing a write method.
    """
    f.write("# ====================\n# | Model parameters |\n# ====================\n#\n")
    headers = [('PARM01', '# Continuous equation parameters'), ('PARM02', '# Elliptic solver parameters'),
               ('PARM03', '# Time stepping parameters'), ('PARM04', '# Gridding parameters'), 
               ('PARM05', '# Input datasets')]
    for sec, header in headers:
        f.write(f'{header}\n &{sec}\n')
        for key, val in data[sec].items():
            f.write(f' {key}={val},\n')
        f.write(' &\n\n')


def open_runfolder(
    folder,
    prefixes="diag",
    merge_full=True,
    partial_prefixes=None,
    exclude=None,
    read_input_data=False,
    verb=False,
):
    """Open a folder and reads all data.

    PARAMETERS
    ----------
    folder   : str or Path
    prefixes : List of prefixes to read in.
               If 'all' : reads all data files with at least 2 iterations
               if '*'   : reads all data files with at least 3 iterations
               if 'diag': reads the data files listed in data.diagnostcs,
                   Files with the same timephase and frequency are merged under the name of the first one.
               To read partial data files (with selection of levels) use 'diag'.
    merge_full : Bool, if True, merges all non-partial-levels datasets under the name "full"
    partial_prefixes : List of (prefix, levels) where levels is a list of indexes
                       Prefixes with partial levels to read separetely.
    exclude  : List of prefixes to exclude.
    read_input_data : Bool, if True, reads all *.bin files in the folder and tries to guess their size.
                            Unable to read files with other dimensions than X, Y and Z.
                            Adds the variables to the first dataset (or the merged one)
    verb     : Bool, whether to print verbose info or not.
    silence_cast_errors : Bool, passed to parse_namelist

    RETURNS
    -------
    datasets : Dict, keys are the prefixes, values the xr.Datasets
    """

    folder = Path(folder)

    data = parse_namelist(folder / "data", flat=True, silence_cast_errors=True)
    data.setdefault('deltat', data.get('deltatclock', data.get('deltatmom')))
    diags = {}
    if prefixes == "diag":
        try:
            diags = parse_namelist(
                folder / "data.diagnostics", silence_cast_errors=True
            )
        except FileNotFoundError:
            warn("File data.diagnostics not found - switching prefixes to all")
            prefixes == "all"

    exclude = exclude or []
    partial_prefixes = partial_prefixes or []
    if verb:
        if exclude:
            print("Ne lira pas {}".format(exclude))
        print("Delta T found: {}".format(data["deltat"]))

    if prefixes == "all" or prefixes == "*":
        all_prefixes = {
            data_file.stem.split(".")[0] for data_file in folder.glob("*.*.data")
        }.difference(
            {
                "pickup",
                "pickup_ptracers",
                *exclude,
                *[prf for prf, _ in partial_prefixes],
            }
        )
        grouped_prefixes = {}
        for prefix in all_prefixes:
            iterations = tuple(
                int(filename.stem.split(".")[-1])
                for filename in sorted(folder.glob(prefix + ".*.data"))
            )
            if len(iterations) > 1 and (len(iterations) > 2 or prefixes == "all"):
                grouped_prefixes.setdefault(iterations, []).append(prefix)
        prefixes = grouped_prefixes.values()
    elif prefixes == "diag":
        grouped_prefixes = {}
        partial_prefixes = []
        for name, lvls, frq, tph in zip_longest(
            diags["DIAGNOSTICS_LIST"]["filename"],
            diags["DIAGNOSTICS_LIST"].get("levels", []),
            diags["DIAGNOSTICS_LIST"]["frequency"],
            diags["DIAGNOSTICS_LIST"]["timephase"],
        ):
            if isinstance(name, str) and name not in exclude:
                if lvls is None or len(lvls) == 0:
                    grouped_prefixes.setdefault((frq, tph), []).append(name)
                else:
                    partial_prefixes.append((name, np.array(lvls, dtype=int)))
        prefixes = grouped_prefixes.values()

    datasets = OrderedDict()
    for prefix in prefixes:
        if verb:
            print("Lecture de {}".format(prefix))
        if isinstance(prefix, list):
            prf = prefix[0]
        else:
            prf = prefix
        datasets[prf] = open_mdsdataset(
            str(folder),
            geometry="cartesian",
            delta_t=data["deltat"],
            prefix=prefix,
            ignore_unknown_vars=True,
        )
        datasets[prf].attrs.update(
            source="Created from files {}*.data/meta".format(prefix)
        )

    if merge_full:
        _, full = datasets.popitem(last=False)
        for dataset in datasets.values():
            full = full.merge(dataset)
        full.attrs['config'] = data
        datasets = {"full": full}

    if len(datasets) > 0:
        first_prf, full = next(iter(datasets.items()))
    else:  # No non-partial datasets, so we read the grid.
        first_prf = "grid"
        datasets[first_prf] = open_mdsdataset(
            str(folder),
            iters=None,
            prefix="",
            geometry="cartesian",
            delta_t=data["deltat"],
            ignore_unknown_vars=True,
        )
        full = datasets[first_prf]

    datasets.update(
        {
            prf: open_partial(
                full, folder, prf, lvls, data["deltat"], geometry="cartesian"
            )
            for prf, lvls in partial_prefixes
        }
    )

    if read_input_data:
        binfiles = folder.glob("*.bin")
        for binfile in binfiles:
            if verb:
                print("Lecture de {}.".format(binfile.stem))
            try:
                bindata = readIEEEbin(str(binfile), (full.YC.size, full.XC.size))
            except ValueError:
                try:
                    bindata = readIEEEbin(
                        str(binfile), (full.Z.size, full.YC.size, full.XC.size)
                    )
                except ValueError:
                    if verb:
                        print(
                            "Incapable de lire {varname}, on saute.".format(
                                varname=binfile.stem
                            )
                        )
                    continue
                else:
                    full = full.assign(
                        {
                            binfile.stem: xr.DataArray(
                                bindata,
                                dims=("XC", "YC", "Z"),
                                coords={"XC": full.XC, "YC": full.YC, "Z": full.Z},
                            )
                        }
                    )
            else:
                full = full.assign(
                    {
                        binfile.stem: xr.DataArray(
                            bindata,
                            dims=("XC", "YC"),
                            coords={"XC": full.XC, "YC": full.YC},
                        )
                    }
                )
        datasets[first_prf] = full

    for dataset in datasets.values():
        for coord in dataset.coords.values():
            if coord.name.startswith("Z"):
                coord.attrs.update(long_name="Depth")

    return datasets


def open_partial(full, folder, prefix, levels, delta_t, geometry="cartesian"):
    """Open a dataset with partial levels.

    PARAMETERS
    ----------
    full   : Dataset with all full coordinates
    folder : str, Folder where the dataset is saved
    prefix : str, name of the data file in folder
    levels : Seq of ints, indices of the levels as written in data.diagnostics
    delta_t : int, timesteps in seconds
    geometry : str, geometry

    RETURNS
    -------
    Dataset, the partial dataset
    Grid   , the xgcm Grid
    """
    d = open_mdsdataset(
        str(folder),
        prefix=prefix,
        delta_t=delta_t,
        geometry=geometry,
        nz=len(levels),
        read_grid=False,
    )
    name_dict = dict(
        i="XC", i_g="XG", j="YC", j_g="YG", k="Z", k_u="Zu", k_l="Zl", k_p1="Zp1"
    )
    d = d.rename(name_dict=name_dict)
    for coord in name_dict.values():
        if coord in ["Z", "Zl"]:
            d[coord] = full[coord].isel({coord: levels})
        elif "Z" in coord:
            d = d.drop(coord)
        else:
            d[coord] = full[coord]
    return d
