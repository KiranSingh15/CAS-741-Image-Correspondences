import config as cfg
import specificationParameters as sp

# get user inputs
user_sel_methods = cfg.get_active_functions()
u_methods = list(user_sel_methods.values())
user_sel_params = cfg.get_chosen_parameters()
u_params = list(user_sel_params.values())
# need to add a method to check each image


# get system limits
ifc_def_methods, method_limits = sp.set_default_methods()
ifc_def_params = sp.set_default_parameters()
def_vals = list(ifc_def_params.values())
# print(def_vals)

# check that the selected method is in bounds
idx = 0
for i in user_sel_methods:
    if user_sel_methods[i] < 0 or user_sel_methods[i] > method_limits[idx]:
        print("error in ", i)
    idx += idx

# check that the selected method configuration is valid
### FUTURE WORK ###


# compare each parameter in the range
param_error_count , err_list = sp.checklimits(def_vals[0],def_vals[1],def_vals[2],def_vals[3],def_vals[4])







