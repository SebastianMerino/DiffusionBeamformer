"""
Script to separate training and testing data
"""
import os
import random
import shutil


input_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\input_id'
input_testing = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\input_testing'
output_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\target_enh'
output_testing = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\target_testing'
percentage = 0.2

input_file_list = sorted(os.listdir(input_folder))
num_testing_files = int(len(input_file_list) * percentage)

# Use random.sample to pick the specified percentage of elements
testing_files = random.sample(input_file_list, num_testing_files)

for file in testing_files:
    source_path = os.path.join(input_folder, file)
    destination_path = os.path.join(input_testing, file)
    shutil.move(source_path, destination_path)

    source_path = os.path.join(output_folder, file)
    destination_path = os.path.join(output_testing, file)
    shutil.move(source_path, destination_path)