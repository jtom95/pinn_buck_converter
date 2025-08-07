import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# import from matlab
from scipy.io import loadmat
from scipy.io import savemat

datapath = Path(
    r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\PIML_Converter\Simulation_data"
)
macrogroup = "original_format"

# check the path is recognized
print(f"Path exists: {datapath.exists()}")

# Print the contents of the directory
print("Directory contents:")
for item in datapath.iterdir():
    print(item.name)

# Load the mat files
loaded_data = {}

for file in datapath.glob("*.mat"):
    # use the number in the filename to create a key
    key = file.stem.split("_")[1]
    print(f"Loading {file.name} as {key}")

    loaded_data[key] = loadmat(file)

print(f"Loaded data keys: {loaded_data.keys()}")


# create a function that saves each key in the loaded_data dict to a h5 file
def save_to_h5(data, key, filename):
    # create a new h5 file
    with h5py.File(filename, "a") as f:
        # create or get the macrogroup group
        if macrogroup in f:
            orig_group = f[macrogroup]
        else:
            orig_group = f.create_group(macrogroup)
            # Add attributes describing the macrogroup
            orig_group.attrs[
                "description"
            ] = """
            From the paper: 
                S. Zhao, Y. Peng, Y. Zhang and H. Wang, "Parameter Estimation of Power Electronic Converters 
                With Physics-Informed Machine Learning," in IEEE Transactions on Power Electronics, vol. 37, 
                no. 10, pp. 11567-11578, Oct. 2022, doi: 10.1109/TPEL.2022.3176468. 
                
            This data contains the original simulation data loaded from MATLAB .mat files
            Each subgroup corresponds to different experiments (Table II): 
                1. Clean Data (0.1)
                2. ADC error (0.1)
                3. Sync error (1.6)
                4. 5 noise (0.8)
                5. 10 noise (2.1)
                6. ADC-Sync-5noise (3.6)
                7. ADC-Sync-10noise (4.9)
            In each subgroup, the data is organized as follows: 
                +	"Current", Shape: (1440, 1), Dtype: float64
                +    "CurrentInput", Shape: (1440, 1), Dtype: float64
                +    "CurrentInputLower", Shape: (1440, 1), Dtype: float64
                +    "Dswitch", Shape: (1440, 1), Dtype: uint8
                +    "DswitchLower", Shape: (1440, 1), Dtype: float64
                +    "DswitchTransform", Shape: (1440, 1), Dtype: uint16
                +    "Rload", Shape: (1440, 1), Dtype: float64
                +    "RloadLower", Shape: (722, 1), Dtype: float64
                +    "Voltage", Shape: (1440, 1), Dtype: float64
                +    "VoltageInput", Shape: (1440, 1), Dtype: float64
                +    "VoltageInputLower", Shape: (1440, 1), Dtype: float64
                +    "dt", Shape: (1440, 1), Dtype: float64
                +    "forwaredBackwaredIndicator", Shape: (1440, 2), Dtype: int16
                +    "res", Shape: (1, 1), Dtype: uint8
                +    "t", Shape: (1440, 1), Dtype: float64
                +    "tLower", Shape: (722, 1), Dtype: float64
            The variables "Current", "CurrentInput", and "CurrentInputLower" are all related. Indeed, "CurrentInput"
            is the initial value of a current before a switching transient, while "Current" is the value it reaches 
            after the transient has finished. However, the initial value of one transient is the final value of the previous one.
            Therefore, the data values are two by two identical, except for the first one, which is the initial value of the first transient.
            The same is true for the variables "Voltage", "VoltageInput", and "VoltageInputLower". 
            
            The dataset "t" is NOT the time variable! instead it is related to each switching transient to finish (unclear exactly what). The variable "dt" is the
            time it takes for the switching transient to finish.
            
            The data used for the experiments is only: 
            - "Current"
            - "CurrentInput"
            - "Voltage"
            - "VoltageInput"
            - "Dswitch"
            - "dt"
            - "forwaredBackwaredIndicator": used for the Runga-Kutta method to determine the direction of the integration.
            The rest of the variables are not used in the experiments.
            
            Each dataset contains doubled data, since the first value of each transient is the final value of the previous one. The data is two-by-two identical, except for the first one, which is 
            the initial value of the first transient, for "CurrentInput" and "VoltageInput". The same is true for the variables "Current" and "Voltage" for the final value of the transient.
            
            However, the datasets also contain different transients corresponding to different load changes: R0->R1, R1->R2, R2->R3:
                + R1: 3.1 Ohm
                + R2: 10.2 Ohm
                + R3: 6.1 Ohm
            The transients are simply concatenated, so the first transient is R0->R1, the second is R1->R2, and the third is R2->R3. This can cause some data to not be identical in pairs. 
             
            You can find the original data in .mat format and the rest of the project at https://github.com/ms140429/PIML_Converter
            """
            orig_group.attrs["source"] = str(datapath)
            orig_group.attrs["created_by"] = "load_simul_data.py script"
            orig_group.attrs["created_on"] = __import__("datetime").datetime.now().isoformat()
        # create a group for the key under macrogroup
        group = orig_group.create_group(key)
        # Add attributes to describe the subgroup
        group.attrs["mat_file"] = str(key)
        group.attrs["source"] = str(datapath / f"{key}.mat")
        # loop through the data and save each item
        for k, v in data.items():
            # check if the value is a numpy array
            if isinstance(v, np.ndarray):
                # save the array to the h5 file
                group.create_dataset(k, data=v)
            else:
                # save the value as an attribute
                group.attrs[k] = v


# save the loaded data to h5 files
db_name = "buck_converter_Shuai.h5"
dir_path = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
db_path = dir_path / db_name

# create the directory if it doesn't exist
dir_path.mkdir(parents=True, exist_ok=True)
# save the data to the h5 file

for key, data in loaded_data.items():
    # create a new h5 file for each key
    filename = db_path.with_name(db_name)
    print(f"Saving {key} to {filename}")

    # save the data to the h5 file
    save_to_h5(data, key, filename)
    print(f"Saved {key} to {filename}")

# Load the h5 file and print the contents
with h5py.File(db_path, "r") as f:
    # print the keys in the file
    print("Keys in the file:")
    for key in f.keys():
        print(key)

    # print the contents of each key
    for key in f.keys():
        print(f"Contents of {key}:")
        for k, v in f[key].items():
            print(f"{k}: {v}")
