import config as cfg
import specificationParameters as sp

# get user inputs
def get_user_inputs():
    user_sel_methods = cfg.get_active_functions()
    user_sel_params = cfg.get_chosen_parameters()

    return user_sel_methods, user_sel_params

# need to add a method to check each image


# get system limits
def get_program_defaults():
    ifc_def_methods, method_limits = sp.set_default_methods()
    ifc_def_params = sp.set_default_parameters()
    def_vals = list(ifc_def_params.values())
    return ifc_def_methods, method_limits ,ifc_def_params


# check that the selected method is in bounds
def check_selected_method(user_sel_methods, method_limits):
    idx = 0
    err_flag = False
    for i in user_sel_methods:
        if user_sel_methods[i] < 0 or user_sel_methods[i] > method_limits[idx]:
            err_flag = True
            print("error in ", i)


    idx += idx
    if err_flag == False:
        print("No errors detected in selected methods.")

# check that the selected method configuration is valid


### FUTURE WORK ###
# compare each parameter in the range


## Main program
def run_input_format_module():
    ifc_def_methods, method_limits, ifc_def_params = get_program_defaults() # check what methods and defaults are available
    user_sel_methods, user_sel_params = get_user_inputs()   # get the user selections

    # get the values in the dictionary as an array
    def_vals = list(ifc_def_params.values())
    u_methods = list(user_sel_methods.values())
    u_params = list(user_sel_params.values())

    # check that the index of the selected method is within bounds
    check_selected_method(user_sel_methods, method_limits)

    # check that the input parameters fall within bounds
    param_error_count , err_list = sp.checklimits(def_vals[0],def_vals[1],def_vals[2],def_vals[3],def_vals[4])

    return u_methods, u_params


u_methods, u_params = run_input_format_module()





