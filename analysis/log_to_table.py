import re

with open(r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\RESNET152_TRAINING LOGS.txt") as file:
    file_str_list = file.read()

for line in file_str_list.split('val_loss'):
    print(line.split(' ')[1].split(';')[0])

for line in file_str_list.split('val_acc'):
    print(line.split(' ')[1].split('\ncurrent')[0])