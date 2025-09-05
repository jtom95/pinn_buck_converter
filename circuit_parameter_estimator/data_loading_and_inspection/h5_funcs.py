from pathlib import Path
import h5py

def explore_h5(filepath: Path | str, indent: bool = True) -> None:
    """Recursively list groups and datasets in an HDF5 file with numbered levels and optional indentation."""

    if isinstance(filepath, str):
        filepath = Path(filepath)

    # check the folder exists
    file_folder = Path(filepath).parent
    if not file_folder.exists():
        raise FileNotFoundError(f"The folder {file_folder} does not exist.")

    # check the file exists
    # allow for .h5 or .hdf5 extensions, if not present add the correct one
    if not Path(filepath).suffix in [".h5", ".hdf5"]:
        filepath = Path(filepath).with_suffix(".h5")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    if not isinstance(filepath, (str, Path)):
        raise TypeError("The filepath must be a string or Path.")

    def recurse(name, obj, depth=1):
        indent_str = ("\t" * (depth - 1)) if indent else ""
        if isinstance(obj, h5py.Group):
            print(f"{indent_str}[{depth}] Group   {name}")
            for key in obj:
                recurse(f"{name}/{key}", obj[key], depth + 1)
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent_str}[{depth}] Dataset {name} - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(filepath, "r") as f:
        for key in f:
            recurse(key, f[key], 1)

