import os

base_dir = './Offline Forgeries'
files = os.listdir(base_dir)

files_dict = {}

for file in files:
    short_file = file[4::]

    class_name = short_file.split('_')[0]

    if class_name not in files_dict:
        files_dict[class_name] = []

    files_dict[class_name].append(file)

for class_name in files_dict:
    cnt = 0
    files = files_dict[class_name]

    for file in files:
        cnt = cnt + 1
        new_name = file[4::].split('_')[0] + '_'

        if cnt < 10:
            new_name = new_name + '0' + str(cnt)
        else:
            new_name = new_name + str(cnt)

        new_name = new_name + '.PNG'

        os.rename(base_dir + '/' + file, 'test/' + new_name)