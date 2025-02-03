"""
Convert all notebooks (files ending with .ipynb) in this folder to Python scripts, do some reformatting and move to
examples folder (overwriting existing files). This way, notebooks are available in the gallery (even though not with
perfect formatting) and are included in the tests.
"""

import os
import subprocess
import shutil
from pathlib import Path
import re

# Do some reformatting
fix_header = True  # this is required for inclusion in gallery
replace_input_cells = True  # this breaks up the code by making input cells visible
replace_bold_comments = True  # this replaces bold comments (lines with # **XXX***) with headings

# Define source and destination folders
source_folder = Path(__file__).parent.resolve()  # Use the folder of the script
destination_folder = source_folder / "../examples/"

# Ensure the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a Jupyter notebook and starts with "Lecture_" or "Practical_"
    if filename.endswith(".ipynb") and (filename.startswith("Lecture_") or
                                        filename.startswith("Practical_") or
                                        filename.startswith("Coursework")):
        source_file = source_folder / filename

        # Convert the notebook to a Python script
        subprocess.run(["jupyter", "nbconvert", "--to", "script", str(source_file)], check=True)

        # Construct the generated Python script filename
        python_file = filename.replace(".ipynb", ".py")
        generated_file = source_folder / python_file

        # Modify the beginning and content of the file
        if generated_file.exists():
            with open(generated_file, "r+") as f:
                lines = f.readlines()
                if fix_header:
                    if len(lines) >= 2 and lines[0].startswith("#!") and lines[1].startswith("# coding: utf-8"):
                        # Look for the first markdown header
                        header = "Example\n======="
                        for line in lines[2:]:
                            if line.startswith("# # "):
                                header = line[3:].strip() + "\n" + "=" * len(line[3:].strip())
                                break
                        lines = [f"\n\"\"\"\n{header}\n\"\"\"\n"] + lines[2:]

                if replace_input_cells:
                    # Replace In[XXX] pattern with # %%
                    lines = [re.sub(r"# In\[\d+\]:", "# %%", line) for line in lines]

                if replace_bold_comments:
                    # Replace lines starting with # **XXX** with formatted block
                    lines = [re.sub(r"# \*\*(.+?)\*\*",
                                    lambda m: f"# %%\n# {m.group(1)}\n# {'-' * len(m.group(1))}\n# \n#", line) for
                             line in lines]

                f.seek(0)
                f.writelines(lines)
                f.truncate()

            # Move the Python script to the destination folder, overwriting if necessary
            destination_file = destination_folder / python_file
            if destination_file.exists():
                destination_file.unlink()
            shutil.move(str(generated_file), str(destination_file))
            print(f"Moved: {python_file} to {destination_folder}")
        else:
            print(f"Failed to find generated file: {python_file}")
