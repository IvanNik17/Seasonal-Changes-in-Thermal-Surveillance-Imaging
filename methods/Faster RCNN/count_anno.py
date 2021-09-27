import os

folders = ['data/harborfront/validation/', 'data/harborfront/train/1 one day in Feb 2021', 'data/harborfront/train/2 one week in Feb 2021', 'data/harborfront/train/3 one month in Feb 2021', 'data/harborfront/test/Apr/', 'data/harborfront/test/Aug/', 'data/harborfront/test/Jan']

images = 0
annotations = 0
for fl in folders:
    labels = os.listdir(fl + '/labels/')
    files = os.listdir(fl + '/images/')
    images += len(files)
    for f in labels:
        label_file = open(os.path.join(fl + '/labels/' + f))
        nonempty_lines = [line.strip("\n") for line in label_file if line != "\n"]
        annotations += len(nonempty_lines)
        label_file.close()

print("images: {}".format(images))
print("annotations: {}".format(annotations))
