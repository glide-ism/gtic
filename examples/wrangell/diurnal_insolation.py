"""
Compute and plot diurnal insolation for a single day on the Wrangell ice cap.

Evaluates instantaneous solar potential every 3 hours and produces a
composite figure showing the spatial pattern at each time step.
"""

import logging

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from gtic import SolarPotential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEOM_PATH = './model_inputs/gridded_dem.nc'
GRID_RESOLUTION = 90.0  # meters
LATITUDE = 61.0
LONGITUDE = -143.0
TIMEZONE = "America/Anchorage"
CLEARSKY_TRANSMITTANCE = 0.7
YEAR = 2011
MONTH = 4
DAY = 21  # summer solstice
HOURS = list(range(0, 24, 3))

# Load DEM
logger.info("Loading DEM from %s", GEOM_PATH)
dem = xr.load_dataset(GEOM_PATH)

# Initialize solar potential calculator
logger.info("Initializing SolarPotential calculator...")
solar = SolarPotential(
    dem=dem,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    grid_resolution=GRID_RESOLUTION,
    timezone=TIMEZONE,
    clearsky_transmittance=CLEARSKY_TRANSMITTANCE,
)

# Compute instantaneous potential at each 3-hour step
logger.info("Computing potential for %d-%02d-%02d at 3-hour intervals", YEAR, MONTH, DAY)
potentials = {}
for hour in HOURS:
    logger.info("  hour %02d:00", hour)
    potentials[hour] = solar.potential(YEAR, MONTH, DAY, hour).get()

# Hillshade for background
z_cpu = solar.z.get()
ls = LightSource(azdeg=315, altdeg=45)
dx = dem.x[1].item() - dem.x[0].item()
hs = ls.hillshade(z_cpu, vert_exag=3, dx=dx, dy=dx)

# Find global max across all hours for a shared colorbar scale
vmax = max(p.max() for p in potentials.values())

# Composite plot: 2 rows x 4 columns
ncols = 4
nrows = len(HOURS) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

for ax, hour in zip(axes.flat, HOURS):
    ax.imshow(hs, cmap=plt.cm.gray)
    im = ax.imshow(potentials[hour], alpha=0.6, cmap=plt.cm.plasma, vmin=0, vmax=vmax)
    ax.set_title(f"{hour:02d}:00")
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f"Diurnal Solar Potential — {YEAR}-{MONTH:02d}-{DAY:02d} ({TIMEZONE})", fontsize=14)
fig.colorbar(im, ax=axes, label="Solar potential (dimensionless)", shrink=0.8)
plt.tight_layout()
plt.show()
