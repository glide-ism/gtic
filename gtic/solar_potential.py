import calendar
import datetime
import logging
from pathlib import Path

import cupy as cp
import numpy as np
import pytz
import xarray as xr
from pysolar.solar import get_altitude, get_azimuth
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SolarPotential:
    """
    GPU-accelerated solar potential calculator using CUDA ray tracing.

    For a given DEM and geographic location, computes per-pixel solar
    potential accounting for terrain self-shadowing and incidence angle.
    """

    # Atmospheric scale height in meters (standard atmosphere)
    SCALE_HEIGHT = 8500.0

    def __init__(
        self,
        dem: xr.Dataset,
        latitude: float,
        longitude: float,
        kernel_path: str = None,
        grid_resolution: float = 90.0,
        step_size: float = 1.0,
        timezone: str = "America/Anchorage",
        clearsky_transmittance: float = None,
    ):
        """
        Parameters
        ----------
        dem : xr.Dataset
            Dataset with an 'elevation' variable on an (y, x) grid.
        latitude : float
            Representative latitude for solar angle calculations.
        longitude : float
            Representative longitude for solar angle calculations.
        kernel_path : str, optional
            Path to the azimuth_trace CUDA kernel source file. If not provided,
            defaults to the bundled kernel in gtic/cuda/azimuth_trace.cu.
        grid_resolution : float
            Grid spacing in meters, used for gradient and zenith calculations.
        step_size : float
            Ray marching step size passed to the CUDA kernel.
        timezone : str
            Timezone string for solar position calculations.
        clearsky_transmittance : float, optional
            Broadband clear-sky atmospheric transmittance per unit air mass.
            Typical values are 0.6-0.8. When set, solar potential is attenuated
            by transmittance^(air_mass * pressure_ratio), where air_mass
            accounts for solar zenith angle (Kasten-Young formula) and
            pressure_ratio accounts for thinner atmosphere at elevation.
            If None (default), no atmospheric attenuation is applied.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.grid_resolution = cp.float32(grid_resolution)
        self.step_size = cp.float32(step_size)
        self.timezone = pytz.timezone(timezone)

        self.z = cp.array(dem.elevation.values, dtype=cp.float32)
        self.ny, self.nx = self.z.shape

        self.clearsky_transmittance = clearsky_transmittance
        if clearsky_transmittance is not None:
            self._pressure_ratio = cp.exp(-self.z / self.SCALE_HEIGHT)

        dZdy, dZdx = cp.gradient(self.z, grid_resolution)
        self.dZdx = dZdx
        self.dZdy = -dZdy

        # Resolve kernel path: use bundled kernel if not provided
        if kernel_path is None:
            module_dir = Path(__file__).parent
            kernel_path = module_dir / "cuda" / "azimuth_trace.cu"
        else:
            kernel_path = Path(kernel_path)

        # Load CUDA kernel
        with open(kernel_path, "r") as f:
            kernel_code = f.read()
        kernels = cp.RawModule(code=kernel_code)
        self.kernel = kernels.get_function("azimuth_trace")
        self.block_size = (16, 16)
        self.grid_size = (self.nx // 16 + 1, self.ny // 16 + 1)

    # ── Private physics helpers ──────────────────────────────────────

    def _run_shadow_kernel(self, azimuth_deg: float):
        """
        Run the ray tracing kernel for a given solar azimuth.

        Returns
        -------
        max_zenith : cp.ndarray
            Maximum terrain zenith angle along each ray, shape (ny, nx).
        """
        max_zenith = cp.zeros(self.z.shape, dtype=cp.float32)
        max_j = cp.zeros(self.z.shape, dtype=cp.uint32)
        max_i = cp.zeros(self.z.shape, dtype=cp.uint32)

        j_basis = cp.float32(np.sin(np.deg2rad(azimuth_deg)))
        i_basis = -cp.float32(np.cos(np.deg2rad(azimuth_deg)))

        self.kernel(
            self.grid_size,
            self.block_size,
            (max_zenith, max_j, max_i, self.z, j_basis, i_basis,
             self.step_size, self.nx, self.ny),
        )
        return max_zenith

    def _zenith_deg(self, azimuth_deg: float) -> cp.ndarray:
        """Terrain horizon zenith angle in degrees. Shape (ny, nx)."""
        max_zenith = self._run_shadow_kernel(azimuth_deg)
        return cp.rad2deg(cp.arctan(max_zenith / self.grid_resolution))

    def _shadow_mask(self, altitude_deg: float, azimuth_deg: float) -> cp.ndarray:
        """Soft shadow mask in [0, 1]. Shape (ny, nx)."""
        zenith_deg = self._zenith_deg(azimuth_deg)
        z_i = altitude_deg - zenith_deg
        return 1.0 / (1.0 + cp.exp(-z_i / 0.1))

    def _incidence(self, altitude_deg: float, azimuth_deg: float) -> cp.ndarray:
        """Cosine of solar incidence angle, clamped to [0, 1]. Shape (ny, nx)."""
        sin_phi = cp.sin(cp.deg2rad(cp.float32(azimuth_deg)))
        cos_phi = cp.cos(cp.deg2rad(cp.float32(azimuth_deg)))
        sin_alpha = cp.sin(cp.deg2rad(cp.float32(altitude_deg)))
        cos_alpha = cp.cos(cp.deg2rad(cp.float32(altitude_deg)))

        incidence = (
            -self.dZdx * sin_phi * cos_alpha
            - self.dZdy * cos_phi * cos_alpha
            + sin_alpha
        ) / cp.sqrt(self.dZdx ** 2 + self.dZdy ** 2 + 1)

        incidence = cp.maximum(incidence, 0.0)
        return incidence

    def _air_mass(self, altitude_deg: float) -> float:
        """Kasten-Young (1989) air mass from solar altitude in degrees."""
        if altitude_deg <= 0:
            return float('inf')
        a = np.deg2rad(altitude_deg)
        return 1.0 / (np.sin(a) + 0.50572 * (altitude_deg + 6.07995) ** -1.6364)

    def _atmospheric_attenuation(self, altitude_deg: float):
        """
        Per-pixel atmospheric transmittance factor.

        Returns 1.0 when clearsky_transmittance is None (no attenuation).
        Otherwise returns transmittance^(air_mass * pressure_ratio),
        shape (ny, nx).
        """
        if self.clearsky_transmittance is None:
            return 1.0
        am = self._air_mass(altitude_deg)
        if np.isinf(am):
            return cp.zeros(self.z.shape, dtype=cp.float32)
        return self.clearsky_transmittance ** (am * self._pressure_ratio)

    def _solar_position(self, dt: datetime.datetime) -> tuple:
        """
        Solar (altitude_deg, azimuth_deg) for a timezone-aware datetime.
        """
        altitude = get_altitude(self.latitude, self.longitude, dt)
        azimuth = get_azimuth(self.latitude, self.longitude, dt)
        return altitude, azimuth

    def _potential_at(self, altitude_deg: float, azimuth_deg: float) -> cp.ndarray:
        """
        Instantaneous solar potential for a single solar position.

        Returns incidence * shadow_mask. Shape (ny, nx), values in [0, 1].
        """
        shadow = self._shadow_mask(altitude_deg, azimuth_deg)
        incidence = self._incidence(altitude_deg, azimuth_deg)
        return incidence * shadow * self._atmospheric_attenuation(altitude_deg)

    # ── Public temporal query interface ──────────────────────────────

    def potential(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
    ) -> cp.ndarray:
        """
        Solar potential at a specific instant.

        Parameters
        ----------
        year, month, day : int
            Date in the configured timezone.
        hour : int
            Hour of day (0-23) in the configured timezone.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
            Instantaneous solar potential in [0, 1].
        """
        dt = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=self.timezone)
        altitude, azimuth = self._solar_position(dt)
        return self._potential_at(altitude, azimuth)

    def potential_daily(
        self,
        year: int,
        month: int,
        day: int,
        reduction: str = "mean",
    ) -> cp.ndarray:
        """
        Solar potential accumulated over all 24 hours of a single day.

        Parameters
        ----------
        year, month, day : int
        reduction : {'mean', 'sum'}
            'mean': divide by 24 for mean hourly intensity.
            'sum': raw accumulation over 24 hourly samples.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
        """
        accumulated = cp.zeros(self.z.shape, dtype=cp.float32)
        for hour in tqdm(range(24), desc=f"{year}-{month:02d}-{day:02d}", unit="hr"):
            dt = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=self.timezone)
            altitude, azimuth = self._solar_position(dt)
            accumulated += self._potential_at(altitude, azimuth)

        if reduction == "mean":
            accumulated /= 24
        return accumulated

    def potential_monthly(
        self,
        year: int,
        month: int,
        reduction: str = "mean",
    ) -> cp.ndarray:
        """
        Solar potential accumulated over all hours of all days in a month.

        Parameters
        ----------
        year, month : int
        reduction : {'mean', 'sum'}
            'mean': divide by (num_days * 24) for mean intensity.
            'sum': raw accumulation.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
        """
        num_days = calendar.monthrange(year, month)[1]
        accumulated = cp.zeros(self.z.shape, dtype=cp.float32)

        for day in tqdm(range(1, num_days + 1), desc=f"month {month:02d}", unit="day"):
            for hour in range(24):
                dt = datetime.datetime(
                    year, month, day, hour, 0, 0, tzinfo=self.timezone
                )
                altitude, azimuth = self._solar_position(dt)
                accumulated += self._potential_at(altitude, azimuth)

        if reduction == "mean":
            accumulated /= num_days * 24
        return accumulated

    def potential_monthly_all(
        self,
        year: int,
        reduction: str = "mean",
    ) -> cp.ndarray:
        """
        Solar potential for all 12 months of a year.

        Parameters
        ----------
        year : int
        reduction : {'mean', 'sum'}

        Returns
        -------
        cp.ndarray, shape (12, ny, nx)
        """
        result = cp.zeros((12, self.ny, self.nx), dtype=cp.float32)
        for month in range(1, 13):
            result[month - 1] = self.potential_monthly(year, month, reduction=reduction)
        return result

    def potential_hourly_by_month(
        self,
        year: int,
        month: int = None,
        reduction: str = "mean",
    ) -> cp.ndarray:
        """
        Hourly solar potential profile averaged over each day in a month.

        For each of the 24 hour-of-day bins, accumulates potential over all
        days in the month, then reduces.

        Parameters
        ----------
        year : int
        month : int or None
            If given, compute for a single month. Returns shape (24, ny, nx).
            If None, compute for all 12 months. Returns shape (12, 24, ny, nx).
        reduction : {'mean', 'sum'}
            'mean': divide each hour-bin by num_days.
            'sum': raw accumulation over days.

        Returns
        -------
        cp.ndarray
            Shape (24, ny, nx) if month is specified,
            shape (12, 24, ny, nx) if month is None.
        """
        if month is not None:
            return self._potential_hourly_single_month(year, month, reduction)

        result = cp.zeros((12, 24, self.ny, self.nx), dtype=cp.float32)
        for m in range(1, 13):
            result[m - 1] = self._potential_hourly_single_month(year, m, reduction)
        return result

    def _potential_hourly_single_month(
        self, year: int, month: int, reduction: str
    ) -> cp.ndarray:
        """Compute hourly profile for a single month. Shape (24, ny, nx)."""
        num_days = calendar.monthrange(year, month)[1]
        hourly = cp.zeros((24, self.ny, self.nx), dtype=cp.float32)

        for day in tqdm(range(1, num_days + 1), desc=f"month {month:02d}", unit="day"):
            for hour in range(24):
                dt = datetime.datetime(
                    year, month, day, hour, 0, 0, tzinfo=self.timezone
                )
                altitude, azimuth = self._solar_position(dt)
                hourly[hour] += self._potential_at(altitude, azimuth)

        if reduction == "mean":
            hourly /= num_days
        return hourly

    def potential_fourier(
        self,
        year: int,
        reduction: str = "mean",
    ) -> tuple:
        """
        Fourier decomposition of the diurnal solar potential cycle by month.

        Compresses the (12, 24, ny, nx) hourly-by-month representation into
        three (12, ny, nx) coefficient fields: the DC component (c0), and
        the first-harmonic cosine (cc) and sine (cs) coefficients.

        The hourly potential for month m at hour h can be reconstructed as:
            potential[m, h] ~ c0[m] + cc[m]*cos(2*pi*h/24) + cs[m]*sin(2*pi*h/24)

        Parameters
        ----------
        year : int
        reduction : {'mean', 'sum'}
            Passed through to potential_hourly_by_month.

        Returns
        -------
        c0, cc, cs : tuple of cp.ndarray
            Each has shape (12, ny, nx).
        """
        result = self.potential_hourly_by_month(year=year, reduction=reduction)
        ny, nx = result.shape[2], result.shape[3]

        c0 = cp.zeros((12, ny, nx), dtype=cp.float32)
        cc = cp.zeros((12, ny, nx), dtype=cp.float32)
        cs = cp.zeros((12, ny, nx), dtype=cp.float32)

        for h in range(24):
            slc = result[:, h, :, :]
            c0 += slc
            cc += slc * cp.cos(2 * np.pi * h / 24)
            cs += slc * cp.sin(2 * np.pi * h / 24)
        c0 /= 24
        cc /= 12  # 2/24
        cs /= 12

        return c0, cc, cs

    def sunlight_hours(
        self,
        year: int,
        month: int,
        reduction: str = "mean",
    ) -> cp.ndarray:
        """
        Accumulated sunlight hours per pixel over a month.

        Unlike the potential methods, this uses only the shadow mask (no
        incidence angle), so it measures duration of direct sunlight
        rather than energy potential.

        Parameters
        ----------
        year, month : int
        reduction : {'mean', 'sum'}
            'mean': mean daily sunlight hours (divide by num_days).
            'sum': total shadow-mask accumulation over the month.

        Returns
        -------
        cp.ndarray, shape (ny, nx)
        """
        num_days = calendar.monthrange(year, month)[1]
        accumulated = cp.zeros(self.z.shape, dtype=cp.float32)

        for day in tqdm(range(1, num_days + 1), desc=f"sunlight month {month:02d}", unit="day"):
            for hour in range(24):
                dt = datetime.datetime(
                    year, month, day, hour, 0, 0, tzinfo=self.timezone
                )
                altitude, azimuth = self._solar_position(dt)
                accumulated += self._shadow_mask(altitude, azimuth)

        if reduction == "mean":
            accumulated /= num_days
        return accumulated
