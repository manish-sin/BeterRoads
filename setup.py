import os
import shutil

# Here we will do some initial setup
# We will remove all the folders and some cvs files which needs to be empty when starting a fresh project
# Currently the folders populated with data for reference
# commands below will remove these folders and create a new one with same name

#Here we define the path of all such folders
Root_dir = os.path.abspath(".")
frames_path = os.path.join(Root_dir, r"frames")
wrong_coordinate_extacrtion_path = os.path.join(Root_dir, r"wrong_coordinate_extacrtion")
road_images_path = os.path.join(Root_dir, r"road_images")
pot_holes_detected_path = os.path.join(Root_dir, r"pot_holes_detected")

#Removing those folders
shutil.rmtree(frames_path)
shutil.rmtree(wrong_coordinate_extacrtion_path )
shutil.rmtree(road_images_path)
shutil.rmtree(pot_holes_detected_path)

#recreating those folders
os.mkdir(frames_path)
os.mkdir(wrong_coordinate_extacrtion_path)
os.mkdir(road_images_path)
os.mkdir(pot_holes_detected_path)

#Removing Files
core_data_path = os.path.join(Root_dir, r"core_data.csv")
os.remove(core_data_path)
super_core_data_path = os.path.join(Root_dir, r"super_core_data.csv")
os.remove(super_core_data_path )
final_csv_path = os.path.join(Root_dir, r"final_csv_data.csv")
os.remove(final_csv_path)