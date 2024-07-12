# cone-db

Repository of NIST Cone Calorimeter Data

# Data format

## Metadata

Empty / undefined values are `null` in the metadata file.

| Name                       | Data type                                                                                       | Description                    |
| -------------------------- |:-----------------------------------------------------------------------------------------------:| ------------------------------ |
| `date`                     | String, [ISO8601](https://en.wikipedia.org/wiki/ISO_8601) formatted, e.g. `2019-01-29T14:11:00` |                                |
| `laboratory`               | String                                                                                          |                                |
| `operator`                 | String                                                                                          |                                |
| `report_name`              | String                                                                                          |                                |
| `pretest_comments`         | String                                                                                          |                                |
| `posttest_comments`        | String                                                                                          |                                |
| `comments`        | String                                                                                          |                                |
| `grid`                     | Boolean                                                                                          |                                |
| `mounting_system`          | String                                                                                          | "Edge frame", "CBUF", etc.     |
| `heat_flux_kw/m^2`         | Number                                                                                          |                                |
| `separation_mm`            | Number                                                                                          |                                |
| `manufacturer`             | String                                                                                          |                                |
| `material_id`              | String                                                                                          |                                |
| `specimen_description`     | String                                                                                          |                                |
| `specimen_number`          | String                                                                                          |                                |
| `specimen_prep`            | String                                                                                          |                                |
| `sponsor`                  | String                                                                                          |                                |
| `thickness_mm`             | Number                                                                                          |                                |
| `surface_area_cm^2`        | Number                                                                                          |                                |
| `time_to_ignition_s`       | Number                                                                                          |                                |
| `time_to_flameout_s`       | Number                                                                                          |                                |
| `test_start_time_s`        | Number                                                                                          |                                |
| `user_eot_time_s`          | Number                                                                                          |                                |
| `mlr_eot_mass_g/m^2`       | Number                                                                                          |                                |
| `eot_criterion`            | String                                                                                          |                                |
| `c_factor`                 | Number                                                                                          | SI units                       |
| `od_correction_factor`     | Number                                                                                          |                                |
| `e_mj/kg`                  | Number                                                                                          |                                |
| `initial_mass_g`           | Number                                                                                          |                                |
| `substrate`                | String                                                                                          |                                |
| `non_scrubbed`             | Boolean                                                                                         |                                |
| `orientation`              | String                                                                                          | "horizontal", "vertical", etc. |
| `duct_diameter_m`          | Number                                                                                          |                                |
| `o2_delay_time_s`          | Number                                                                                          |                                |
| `co2_delay_time_s`         | Number                                                                                          |                                |
| `co_delay_time_s`          | Number                                                                                          |                                |
| `ambient_temp_c`           | Number                                                                                          |                                |
| `barometric_pressure_pa`   | Number                                                                                          |                                |
| `relative_humidity_%`      | Number                                                                                          |                                |
| `conditioned`              | Boolean                                                                                         |                                |
| `data_smoothed`            | Boolean                                                                                         |                                |
| `correct_o2_for_pressure`  | Boolean                                                                                         |                                |
| `co_co2_data_collected`    | Boolean                                                                                         |                                |
| `mass_data_collected`      | Boolean                                                                                         |                                |
| `smoke_data_collected`     | Boolean                                                                                         |                                |
| `soot_mass_data_collected` | Boolean                                                                                         |                                |
| `soot_mass_g`              | Number                                                                                          |                                |
| `soot_mass_ratio`          | Number                                                                                          | 1:x                            |

### Events

TBD
