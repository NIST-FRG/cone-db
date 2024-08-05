# cone-db

Repository of NIST Cone Calorimeter Data

# Getting started

## Installing packages
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Data

Raw data should be placed in the `data/raw` folder; folder structure doesn't matter. Run the scripts from the root of the repository, `parse-FTT.py` for FTT-formatted files and `parse-MIDAS.py` for MIDAS-formatted files. HRR, mass flow rate, and Ksmoke (if available) data is calculated from the raw/scaled input data.

Files in the standardized format are saved in the `data/auto-processed` folder.

# Manual review
See [scripts/cone-explorer/README.md](scripts/cone-explorer/README.md) for information on manually reviewing data.

# Data format

## Metadata

Empty / undefined values are `null` in the metadata file.

| Name                      | Data type                                                                                       | Description                                                                                   |
| ------------------------- |:-----------------------------------------------------------------------------------------------:| --------------------------------------------------------------------------------------------- |
| `date`                    | String, [ISO8601](https://en.wikipedia.org/wiki/ISO_8601) formatted, e.g. `2019-01-29T14:11:00` |                                                                                               |
| `material_id`             | String                                                                                          | Format: `<material_name>:<report_name>`, see [process/README.md](process/README.md) for more. |
| `laboratory`              | String                                                                                          |                                                                                               |
| `operator`                | String                                                                                          |                                                                                               |
| `report_name`             | String                                                                                          |                                                                                               |
| `comments`                | String                                                                                          | Pre-test and post-test comments are combined into one field.                                  |
| `grid`                    | Boolean                                                                                         |                                                                                               |
| `mounting_system`         | String                                                                                          | e.g. "Edge frame"                                                                             |
| `heat_flux_kW/m2`         | Number                                                                                          |                                                                                               |
| `separation_mm`           | Number                                                                                          |                                                                                               |
| `manufacturer`            | String                                                                                          |                                                                                               |
| `specimen_description`    | String                                                                                          |                                                                                               |
| `specimen_number`         | String                                                                                          |                                                                                               |
| `specimen_prep`           | String                                                                                          |                                                                                               |
| `sponsor`                 | String                                                                                          |                                                                                               |
| `thickness_mm`            | Number                                                                                          |                                                                                               |
| `surface_area_cm2`        | Number                                                                                          |                                                                                               |
| `time_to_ignition_s`      | Number                                                                                          |                                                                                               |
| `time_to_flameout_s`      | Number                                                                                          |                                                                                               |
| `test_start_time_s`       | Number                                                                                          |                                                                                               |
| `test_end_time_s`         | Number                                                                                          |                                                                                               |
| `mlr_eot_mass_g/m2`       | Number                                                                                          |                                                                                               |
| `eot_criterion`           | String                                                                                          |                                                                                               |
| `c_factor`                | Number                                                                                          | SI units                                                                                      |
| `od_correction_factor`    | Number                                                                                          |                                                                                               |
| `e_mj/kg`                 | Number                                                                                          |                                                                                               |
| `initial_mass_g`          | Number                                                                                          |                                                                                               |
| `substrate`               | String                                                                                          |                                                                                               |
| `non_scrubbed`            | Boolean                                                                                         |                                                                                               |
| `orientation`             | String                                                                                          | "horizontal", "vertical", etc.                                                                |
| `duct_diameter_m`         | Number                                                                                          |                                                                                               |
| `o2_delay_time_s`         | Number                                                                                          |                                                                                               |
| `co2_delay_time_s`        | Number                                                                                          |                                                                                               |
| `co_delay_time_s`         | Number                                                                                          |                                                                                               |
| `ambient_temp_c`          | Number                                                                                          |                                                                                               |
| `barometric_pressure_pa`  | Number                                                                                          |                                                                                               |
| `relative_humidity_%`     | Number                                                                                          |                                                                                               |
| `correct_o2_for_pressure` | Boolean                                                                                         |                                                                                               |
| `co_co2_data_collected`   | Boolean                                                                                         |                                                                                               |
| `mass_data_collected`     | Boolean                                                                                         |                                                                                               |
| `smoke_data_collected`    | Boolean                                                                                         |                                                                                               |
| `events`                  | Array                                                                                           | See below                                                                                     |

### Events

Events are stored as a list of JSON objects, each with the following properties:

| Name    | Data type | Notes                                               |
| ------- |:---------:| --------------------------------------------------- |
| `time`  | Number    | Time of the event                                   |
| `event` | String    | Name of the event (e.g. "Ignition" or "Start Test") |

*Note that data is shifted to remove the start time - for example, if the start time is 50 seconds, then the first 50 seconds are used to calculate the baseline, then removed so that t=50 is now t=0.*
