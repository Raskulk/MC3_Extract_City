Python scripts for extracting cities from Midnight Club 3 (PS2)

How to use: You need the PS2 version of Midnight Club 3.

To extract the city, you need the main city .pck file, the .ppf file with the matching name, and the props file, all located in the same folder. For example: atlanta_midnight_clear.pck, atlanta_midnight_clear.ppf, and atlanta_midnight_clear_props.pck.

Run the script using: python mc3_extract_models.py name_.pck --props name_props.pck
Example: python mc3_extract_models.py atlanta_midnight_clear.pck --props atlanta_midnight_clear_props.pck

Once run, all city models and textures will be extracted.

To import the city into Blender: run the Import_City_Blender script in the Scripting section, making sure to first specify the folder containing the extracted files in the script's OUTPUT_DIR variable. The city will be imported into Blender within a few minutes.
