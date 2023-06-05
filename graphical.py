from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
from tkinter.messagebox import askokcancel
from ase import Atoms
from typing import Literal
from numpy import ndarray


def gui_get_path(is_file: bool = True)\
        -> Path:
    """Get a path via a native GUI; is_file sets whether to return the path to a file (True) or a directory (False)."""
    root = Tk()
    # Using root.after to ensure the window pops in front of editors
    root.after(300, root.focus_force)
    root.after(333, root.withdraw)
    if is_file:
        p = Path(askopenfilename(parent=root))
    else:  # is_directory
        p = Path(askdirectory(parent=root))
    return p


def gui_open(ffmt: Literal["cif", "extxyz", "xyz", "vasp", "poscar"])\
        -> Atoms | list[Atoms]:
    root = Tk()
    # Using root.after to ensure the window pops in front of editors
    root.after(300, root.focus_force)
    root.after(333, root.withdraw)

    fname = Path(askopenfilename(parent=root))
    with open(fname, "r") as file:
        # I'm not going to alias every possible format that ase.io can read here, just the ones I use
        # It will be very easy to add file formats here as the need arises
        match ffmt:
            case "cif":
                from ase.io import cif
                cif_block = list(cif.read_cif(file, slice(None)))  # Returns a generator!
                if len(cif_block) == 1:
                    atoms = cif_block[0]  # Unwrap if there's only one structure in the cif
                else:
                    atoms = cif_block  # If there are multiple structures in the file, just return the list
                return atoms
            case "extxyz" | "xyz":
                from ase.io import extxyz
                atoms = extxyz.read_extxyz(file)
            case "vasp" | "poscar":
                from ase.io import vasp
                vasp.read_vasp(file)
            case _:
                raise ValueError("Unsupported file format")
    return atoms


def gui_savetiff(img: ndarray,
                 default_fname: str | None = None)\
        -> None:
    from tifffile import imwrite
    root = Tk()
    # Using root.after to ensure the window pops in front of editors
    root.after(300, root.focus_force)
    root.after(333, root.withdraw)

    fname = asksaveasfilename(confirmoverwrite=True, parent=root, initialfile=default_fname)
    imwrite(fname, img, photometric='minisblack')


def gui_wait_for_user(title: str | None = None,
                      msg: str | None = None)\
        -> bool:
    root = Tk()
    # Using root.after to ensure the window pops in front of editors
    root.after(300, root.focus_force)
    root.after(333, root.withdraw)

    if title is None:
        title = "Continue"
    if msg is None:
        msg = "Press OK to continue"

    ans = askokcancel(title=title, message=msg, parent=root)
    return ans
