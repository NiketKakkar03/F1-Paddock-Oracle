"""
Column schemas shared between data ingestion and the Race Data Module.
Both fetch_races.py and the runtime Race Data Module import from here.
"""

LAPS_COLUMNS = [
    "lap_number",       # int: lap number (1-indexed)
    "driver_code",      # str: 3-letter driver code (e.g. "VER")
    "team",             # str: constructor name
    "lap_time_s",       # float: lap time in seconds (NaN if no valid time)
    "compound",         # str: tyre compound ("SOFT", "MEDIUM", "HARD", "INTER", "WET")
    "tyre_life",        # int: number of laps on current set
    "position",         # int: track position at end of lap
    "gap_to_leader_s",  # float: gap to race leader in seconds (0.0 for leader)
    "pit_in",           # bool: True if driver pitted at end of this lap
    "pit_out",          # bool: True if driver joined from pit at start of this lap
    "sc_active",        # bool: True if safety car was active during this lap
]

WEATHER_COLUMNS = [
    "lap_number",       # int: lap number (derived from session time)
    "air_temp_c",       # float: air temperature in Celsius
    "track_temp_c",     # float: track temperature in Celsius
    "rainfall",         # bool: True if rain recorded during this lap
    "humidity",         # float: relative humidity percentage
    "wind_speed_ms",    # float: wind speed in m/s
]

LAPS_DTYPES = {
    "lap_number": "int32",
    "driver_code": "str",
    "team": "str",
    "lap_time_s": "float64",
    "compound": "str",
    "tyre_life": "int32",
    "position": "int32",
    "gap_to_leader_s": "float64",
    "pit_in": "bool",
    "pit_out": "bool",
    "sc_active": "bool",
}
