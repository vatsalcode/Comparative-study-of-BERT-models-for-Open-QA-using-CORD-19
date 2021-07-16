import os, os.path, shutil

folder_paths = ["comm_use_subset/pdf_json", "comm_use_subset/pmc_json",
               "noncomm_use_subset/pdf_json", "noncomm_use_subset/pmc_json",
               "custom_license/pdf_json", "custom_license/pmc_json",
               "biorxiv_medrxiv/pdf_json"]
total_count = 0
for f in folder_paths:
  try:
    num_files = os.listdir(f)
    number_of_files = len(num_files)
    print(f, number_of_files)
    total_count += number_of_files
  except Exception as e:
    print(e)
print('Total number of files : ', total_count)
