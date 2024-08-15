import numpy as np
import glob


result_list = glob.glob("detect/" + "*.txt")

electrode_correct_number = 0
electrode_totally_correct_number = 0
electrode_total_number = 90

needle_correct_number = 0
needle_totally_correct_number = 0
needle_total_number = 90

for result_filename in result_list:
    file = open(result_filename)
    support_class = 0
    for line in file:
        content = line.split(":")
        if support_class == 0:
            if content[0] == "support class":
                support_class = content[-1].split("\n")[0]
        elif support_class == "1":
            if content[0] == "correct number":
                correct_number = content[-1].split("\n")[0]
                electrode_correct_number += int(correct_number)
            elif content[0] == "totally correct number":
                totally_correct_number = content[-1].split("\n")[0]
                electrode_totally_correct_number += int(totally_correct_number)
                if totally_correct_number != '1':
                    print("electrode wrong image: ", result_filename)
        elif support_class == "2":
            if content[0] == "correct number":
                correct_number = content[-1].split("\n")[0]
                needle_correct_number += int(correct_number)
            elif content[0] == "totally correct number":
                totally_correct_number = content[-1].split("\n")[0]
                needle_totally_correct_number += int(totally_correct_number)
                if totally_correct_number != '1':
                    print("needle wrong image: ", result_filename)

print("electrode correct number: ", electrode_totally_correct_number)
print("needle correct number: ", needle_totally_correct_number)
