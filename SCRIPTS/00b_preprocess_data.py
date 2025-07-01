# %% [markdown]
# ## Analysis of Buck Converter Data
#
# The simulation data was saved as a hdf5 database in the folder: C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases.

# %%
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

from pinn_buck.h5_funcs import explore_h5

def separate_transients(
    input_quantity: np.ndarray, 
) -> list:
    """
    Separates the input quantity into subtransients.
    """
    subtransients = []
    total_length = len(input_quantity)
    if not total_length % 3 == 0:
        raise ValueError("The length of the input quantity must be a multiple of 3.")
    subtransient_length = total_length // 3
    for i in range(3):
        start_idx = i * subtransient_length
        end_idx = start_idx + subtransient_length
        subtransients.append(input_quantity[start_idx:end_idx])
    return subtransients


def process_data(
    db_dir: Path, 
    db_name: str,
    output_filename: str, 
    group_number: int = 0
) -> None:
    """
    Processes the data from the hdf5 file and saves it to a new file.
    
    Parameters:
        db_dir (Path): The db_dir where the hdf5 file is located.
        db_name (str): The name of the hdf5 file without extension.
        group_number (int): The group number to process.
    """


    GROUP_NUMBER_DICT = {
        0: "ideal",
        1: "ADC_error",
        2: "Sync Error",
        3: "5 noise",
        4: "10 noise",
        5: "ADC-Sync-5noise",
        6: "ADC-Sync-10noise",
    }

    output_file = db_dir / f"{db_name}_processed.h5"

    with h5py.File(db_dir / f"{db_name}.h5", "r") as f:
        group = f[f"original_format/{group_number}"]

        # i_L datasets
        i_: np.ndarray = group["Current"][:].squeeze()
        i_in: np.ndarray = group["CurrentInput"][:].squeeze()

        # v datasets
        v_: np.ndarray = group["Voltage"][:].squeeze()
        v_in: np.ndarray = group["VoltageInput"][:].squeeze()

        # t datasets
        dt: np.ndarray = group["dt"][:].squeeze()

        # d switch datasets
        d_switch: np.ndarray = group["Dswitch"][:].squeeze().astype(np.int32)

    v_in_subtransients = separate_transients(v_in)
    v_subtransients = separate_transients(v_)

    i_in_subtransients = separate_transients(i_in)
    i_subtransients = separate_transients(i_)

    d_switch_subtransients = separate_transients(d_switch)
    dt_subtransients = separate_transients(dt)

    v_in_subtransients = list(map(lambda x: x[1:-1:2], v_in_subtransients))
    v_subtransients = list(map(lambda x: x[1:-1:2], v_subtransients))
    i_in_subtransients = list(map(lambda x: x[1:-1:2], i_in_subtransients))
    i_subtransients = list(map(lambda x: x[1:-1:2], i_subtransients))
    d_switch_subtransients = list(map(lambda x: x[1:-1:2], d_switch_subtransients))
    dt_subtransients = list(map(lambda x: x[1:-1:2], dt_subtransients))

    # ## Recover Original Signals
    v = v_in_subtransients
    i = i_in_subtransients

    # we can also recover the time vector by cumulating the dt values.
    t_subtransients = [np.cumsum(dt_sub) for dt_sub in dt_subtransients]

    ## Save the final quantities to a new HDF5 file
    output_file = db_dir / output_filename

    # save the different transients as different groups under the macrogroup "ideal"

    with h5py.File(output_file, "a") as f:
        # check if the macrogroup already exists, if not, create it
        if GROUP_NUMBER_DICT[group_number] in f:
            answer = input(f"The file {output_file} already exists. Do you want to overwrite it? (y/n): ")
            if answer.lower() != 'y':
                print(f"File not saved to group {GROUP_NUMBER_DICT[group_number]}.")
                return
            # remove the existing group
            del f[GROUP_NUMBER_DICT[group_number]]
            
            
        macrogroup = f.create_group(GROUP_NUMBER_DICT[group_number])

        for ii, (t_val, v_val, i_val, d_switch_val, dt_val) in enumerate(
            zip(t_subtransients, v, i, d_switch_subtransients, dt_subtransients)
        ):
            # note that v0 and i0 are floats, while the rest are arrays
            subgroup = macrogroup.create_group(f"subtransient_{ii+1}")
            subgroup.create_dataset("t", data=t_val)
            subgroup.create_dataset("v", data=v_val)
            subgroup.create_dataset("i", data=i_val)
            subgroup.create_dataset("Dswitch", data=d_switch_val)
            subgroup.create_dataset("dt", data=dt_val)
            
        print(f"Processed data saved to {output_file}")



if __name__ == "__main__":
    db_dir = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
    db_name = "buck_converter_Shuai"
    output_file_name = f"{db_name}_processed.h5"


    for group_number in range(7):
        process_data(db_dir, db_name, output_file_name, group_number)