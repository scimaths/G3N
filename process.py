import os

for files in os.listdir("logs/tu_classification"):
    vals = open("logs/tu_classification/" + files).read().strip().split('\n')
    max_val = 0
    max_vals = None
    for line in vals:
        if len(line.split(" ")) == 2:
                # print(line.split(" "))
            if float(line.split(" ")[0]) > max_val:
                    max_vals = line.split(" ")
    print(files, max_vals)
 