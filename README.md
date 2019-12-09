# MITgcmtools-py
Hand crafted python tools to play with the MITgcm's output. Not a coherent package, but this might help others.

This repo and "package" contains three modules. I tried to stay general when writing them, but they might be quite specific to my configuration. Example : my model was cartesian, I haven't tested any of this with other geometries.

## `gendata.py`

This file defines functions and a cli to help generate input data of simple shape for the MITgcm. The way I used it is the following: define a `DataPreset`, `SurfacePreset` or `TimePatternPreset` then call it through the `gendata` cli in a bash script that was called just before running `mitgcmuv`. I found this system more scalable than transporting a `gendata.m` file in each of my run folders, as the main shape generation is centralized and version-controlled.

I have written all available "presets" to work on normalized coordinates (0 .. 1). One can see the available presets by calling `gendata list`.

## `io.py`

Defines helper functions to open full run folders in one call (through `xmitgcm`). Also, adds a functionality to read partial levels (when `diagnostics` is used with a `levels`  field), but `xmitgcm` is in the process of implementing this.

## `quantities.py`

Defines a lot of the diagnostics I computed from the raw output of the model (I was always using `diagnostics`). Also provides a cli called `diagnose` that is sadly undertested, but was created in order to do this computation direclty of the computation server, so to be able to copy only a subset of the data. "quantities" are registered with a special decorator so that `diagnose` is aware of them. Those that can use `xgcm` for coordinate handling. Finally, a power density spectrum calculation once was working but moving to a different configuration broke it and it hasn't be repaired. Sorry.

## Other files

`common.py` defines a few things I use (or not) in the other files. `jmd95.py` replicates the Jacket and McDougall (1995) density formula that is used in the MITgcm. This file is directly a syntax-modified copy of the python function available in `MITgcmutils` that ships with the MITgcm.