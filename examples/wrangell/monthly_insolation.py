"""
Compute monthly solar potential for Wrangell ice cap.

Uses gtic's SolarPotential class to compute terrain-corrected insolation
accounting for slope, aspect, and self-shadowing from the DEM.
Saves results to NetCDF.
"""

import logging

import cupy as cp
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from gtic import SolarPotential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VISUALIZE = True
GEOM_PATH = './model_inputs/gridded_dem.nc'
OUTPUT_PATH = './gridded_insolation.nc'
GRID_RESOLUTION = 90.0  # meters
LATITUDE = 61.0
LONGITUDE = -143.0
TIMEZONE = "America/Anchorage"
CLEARSKY_TRANSMITTANCE = 0.7
YEAR = 2011

# Load DEM
logger.info("Loading DEM from %s", GEOM_PATH)
dem = xr.load_dataset(GEOM_PATH)
logger.info("DEM shape: %s", dem.elevation.shape)

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

solar_potential = solar.potential_monthly_all(YEAR)

solar_potential_da = xr.DataArray(
    solar_potential.get(),
    dims=["month", "y", "x"],
    coords={
        "month": np.arange(0, 12, dtype=np.float32),
        "y": dem.y,
        "x": dem.x,
    },
    attrs={
        "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
        "long_name": "monthly solar potential (incidence-weighted, shadow-masked)",
    }
)

# Create output Dataset by copying DEM and adding new variables
out_ds = dem.copy()
out_ds["monthly_solar_potential"] = solar_potential_da

# Add global attributes
out_ds.attrs["source"] = f"gtic SolarPotential calculator, {YEAR}"
out_ds.attrs["location"] = f"Wrangell Ice Cap (lat={LATITUDE}, lon={LONGITUDE})"
out_ds.attrs["grid_resolution"] = f"{GRID_RESOLUTION} m"

del out_ds['elevation']
del out_ds['domain_mask']
del out_ds['rgi_mask']

# Save to NetCDF
logger.info("Saving to %s", OUTPUT_PATH)
out_ds.to_netcdf(OUTPUT_PATH)
logger.info("Saved %s", OUTPUT_PATH)

# Visualization
logger.info("Creating visualization...")
z_cpu = solar.z.get()
ls = LightSource(azdeg=315, altdeg=45)
dx = dem.x[1].item() - dem.x[0].item()
hs = ls.hillshade(z_cpu, vert_exag=3, dx=dx, dy=dx)

# Plot April (month 3, 0-indexed) for visualization
april_idx = 3

# Solar potential overlay
plt.imshow(hs, cmap=plt.cm.gray)
im1 = plt.imshow(solar_potential[april_idx].get(), alpha=0.5, cmap=plt.cm.plasma)
plt.title('April: Daily Average Solar Potential')
plt.colorbar(im1, label='Incidence-weighted (dimensionless)')
plt.tight_layout()
plt.show()
