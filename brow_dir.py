path = "D:/test-wed2"
list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.JPG'): 
            list_of_files[filename] = os.sep.join([dirpath, filename])
