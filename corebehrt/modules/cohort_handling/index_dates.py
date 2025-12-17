import logging
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.preparation.filter import filter_table_by_pids

logger = logging.getLogger(__name__)


class IndexDateHandler:
    @staticmethod
    def create_timestamp_series(pids: Set[str], timestamp: datetime) -> pd.Series:
        """Create a timestamp series for given PIDs."""
        return pd.Series(
            data=timestamp,
            index=pd.Index(list(pids), name=PID_COL),
            name=TIMESTAMP_COL,
        )

    @staticmethod
    def get_index_timestamps_for_exposed(
        pids: Set[str], n_hours_from_exposure: int, exposures: pd.DataFrame
    ) -> pd.Series:
        """Get index timestamps for exposed patients."""
        hours_delta = pd.Timedelta(hours=n_hours_from_exposure)
        exposures = filter_table_by_pids(exposures, pids)
        result = exposures.set_index(PID_COL)[TIMESTAMP_COL] + hours_delta
        result.index.name = PID_COL
        return result

    @staticmethod
    def _ensure_series_format(data: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure data is in Series format with PID as index."""
        if isinstance(data, pd.DataFrame):
            return data.set_index(PID_COL)[TIMESTAMP_COL]
        return data

    @staticmethod
    def draw_index_dates_for_unexposed(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
    ) -> pd.Series:
        """
        Draw censor dates for patients not in censoring_timestamps.
        Includes validation against minimum/maximum index dates.
        """
        np.random.seed(42)

        # Ensure censoring_timestamps is a Series
        censoring_timestamps = IndexDateHandler._ensure_series_format(
            censoring_timestamps
        )

        missing_pids = set(data_pids) - set(censoring_timestamps.index)

        # Draw random timestamps for missing patients
        random_abspos = np.random.choice(
            censoring_timestamps.values, size=len(missing_pids)
        )
        new_entries = pd.Series(
            random_abspos, index=pd.Index(list(missing_pids), name=PID_COL)
        )
        result = pd.concat([censoring_timestamps, new_entries])
        result.index.name = PID_COL

        return result

    @classmethod
    def determine_index_dates(
        cls,
        patients_info: pd.DataFrame,
        index_date_mode: str,
        *,  # force keyword arguments
        absolute_timestamp: Optional[dict] = None,
        n_hours_from_exposure: Optional[int] = None,
        exposures: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Determine index dates based on mode.

        Args:
            patients_info: DataFrame with patients info
            index_date_mode: "absolute" or "relative"
            absolute_timestamp: dict with year, month, day (required if mode == "absolute")
            n_hours_from_exposure: int (required if mode == "relative")
            exposures: DataFrame (required if mode == "relative")

        Returns:
            pd.Series: Index dates for all patients
        """
        pids = set(patients_info[PID_COL].unique())

        if index_date_mode == "absolute":
            absolute_timestamp = datetime(**absolute_timestamp)
            result = cls.create_timestamp_series(pids, absolute_timestamp)
        elif index_date_mode == "relative":
            result = cls._handle_relative_mode(
                pids,
                n_hours_from_exposure,
                exposures,
            )
        else:
            raise ValueError(f"Unsupported index date mode: {index_date_mode}")

        result.index.name = PID_COL
        result.name = TIMESTAMP_COL
        return result

    @classmethod
    def _handle_relative_mode(
        cls,
        pids: Set[str],
        n_hours_from_exposure: Optional[int],
        exposures: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Handle relative mode index date calculation."""
        n_hours = n_hours_from_exposure or 0
        exposed_timestamps = cls.get_index_timestamps_for_exposed(
            pids, n_hours, exposures
        )

        return cls.draw_index_dates_for_unexposed(pids, exposed_timestamps)
