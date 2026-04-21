# gtic

GPU-accelerated topographic insolation calculator.

**gtic** computes per-pixel solar potential over a digital elevation model (DEM),
accounting for terrain self-shadowing and surface incidence angle. A CUDA ray-tracing
kernel marches rays along each solar azimuth to find horizon obstruction, while
surface normals derived from the DEM gradient determine the cosine incidence
correction. Solar positions are computed with [pysolar](https://pysolar.readthedocs.io/).

## Features

- CUDA ray tracing with bilinear DEM interpolation for shadow detection
- Incidence-angle correction from slope and aspect
- Monthly, hourly, and Fourier-decomposed output modes
- Soft shadow transitions via sigmoid masking
- Optional atmospheric attenuation (elevation-dependent pressure and air-mass path length)

## Requirements

- Python >= 3.9
- A CUDA-capable GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CuPy](https://cupy.dev/) (compiled against your CUDA version)

## Installation

```bash
pip install .
```

CuPy must match your local CUDA version. For example, with CUDA 12.x:

```bash
pip install cupy-cuda12x
pip install .
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for details.

## Quick start

```python
import xarray as xr
from gtic import SolarPotential

dem = xr.load_dataset("gridded_dem.nc")

solar = SolarPotential(
    dem=dem,
    latitude=61.0,
    longitude=-143.0,
    grid_resolution=90.0,
    timezone="America/Anchorage",
)

# Instantaneous potential at a single hour
pot = solar.potential(2011, 6, 21, 12)

# Daily average for the summer solstice
daily = solar.potential_daily(2011, 6, 21)

# Monthly mean daily solar potential (12, ny, nx) on GPU
monthly = solar.potential_monthly_all(year=2011)

# Fourier decomposition for compact storage (three (12, ny, nx) arrays)
c0, cc, cs = solar.potential_fourier(year=2011)
```

## API

### `SolarPotential(dem, latitude, longitude, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dem` | `xr.Dataset` | required | Dataset with an `elevation` variable on a (y, x) grid |
| `latitude` | `float` | required | Representative latitude for solar angle calculations |
| `longitude` | `float` | required | Representative longitude for solar angle calculations |
| `kernel_path` | `str` | bundled | Path to a custom CUDA kernel source file |
| `grid_resolution` | `float` | `90.0` | Grid spacing in meters |
| `step_size` | `float` | `1.0` | Ray marching step size (grid cells) |
| `timezone` | `str` | `"America/Anchorage"` | Timezone for solar position calculations |
| `clearsky_transmittance` | `float` | `None` | Broadband clear-sky transmittance per unit air mass (typical 0.6-0.8). Enables atmospheric attenuation when set. |

### Methods

All temporal methods accept `reduction='mean'` (default) or `reduction='sum'`.
With `'mean'`, values are dimensionless intensities relative to continuous orthogonal
sunlight. With `'sum'`, values are raw accumulations over the time steps.

| Method | Returns | Description |
|---|---|---|
| `potential(year, month, day, hour)` | `(ny, nx)` | Instantaneous solar potential at a single hour |
| `potential_daily(year, month, day)` | `(ny, nx)` | Solar potential averaged over 24 hours |
| `potential_monthly(year, month)` | `(ny, nx)` | Solar potential averaged over all hours of a month |
| `potential_monthly_all(year)` | `(12, ny, nx)` | Monthly averages for all 12 months |
| `potential_hourly_by_month(year, month=None)` | `(24, ny, nx)` or `(12, 24, ny, nx)` | Diurnal cycle averaged over each month |
| `potential_fourier(year)` | 3 x `(12, ny, nx)` | Fourier decomposition (mean, cosine, sine) |
| `sunlight_hours(year, month)` | `(ny, nx)` | Shadow-mask only; hours of direct sunlight |

All arrays are returned on-GPU as CuPy arrays. Call `.get()` to transfer to NumPy.

## Example

A full example computing hourly insolation for the Wrangell Ice Cap is in
[`examples/wrangell/`](examples/wrangell/).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
