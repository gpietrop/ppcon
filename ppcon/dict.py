"""
this file contains information regarding the profile discretization of the depth
"""

dict_max_pressure = {"NITRATE": 1000,
                     "CHLA": 200,
                     "BBP700": 200}
dict_interval = {"NITRATE": 5,
                 "CHLA": 1,
                 "BBP700": 1}

dict_var_name = {"NITRATE": "Nitrate",
                 "CHLA": "Chlorophyll",
                 "BBP700": "BBP700"}
dict_unit_measure = {"NITRATE": r"$mmol \ m^{-3}$",
                     "CHLA": r"$mg \ m^{-3}$",
                     "BBP700": r"$m^{-1}$"}

dict_models = {
    "NITRATE": ["2023-12-16", 100],
    "CHLA": ["2023-12-17", 150],
    "BBP700": ["2023-12-15", 125]
}