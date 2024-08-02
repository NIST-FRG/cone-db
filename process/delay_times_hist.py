import json
from pathlib import Path

import plotly.express as px

# Read in all the metadata files
all_metadata_files = list(Path("./data").rglob("*.json"))

# Read in all the delay times
delay_times = []

for test in all_metadata_files:
    with open(test) as f:
        to_read = "o2_delay_time_s"
        data = json.load(f)

        delay_times.append(data[to_read])

# Plot the delay times on a histogram
fig = px.histogram(delay_times)

# add x & y labels
fig.update_layout(xaxis_title="O2 delay time (s)", yaxis_title="Frequency")

fig.show()
