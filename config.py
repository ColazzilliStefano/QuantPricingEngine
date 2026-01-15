import os

# Output directory
CHART_DIR = "charts"
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# Data Fetching Configuration
SERIES_IDS = {
    '3M': 'DGS3MO', '6M': 'DGS6MO', '1Y': 'DGS1', '2Y': 'DGS2',
    '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7', '10Y': 'DGS10',
    '30Y': 'DGS30'
}

MATURITY_MAP = {
    '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0, '3Y': 3.0, '5Y': 5.0,
    '7Y': 7.0, '10Y': 10.0, '30Y': 30.0
}

START_DATE = '2020-01-01'

# Optimization Bounds
PARAM_BOUNDS = (
    (0.0, 0.15),     # b0
    (-0.15, 0.15),   # b1
    (-0.20, 0.30),   # b2
    (-0.20, 0.30),   # b3
    (0.1, 10.0),     # l1
    (0.5, 25.0)      # l2
)
