import matplotlib.pyplot as plt
import csv

x = []
y = []
foldername = "/home/anton/PycharmProjects/Hierarchical-Federated-Learning/results/"
filenames = [
    "run-train-tag-epoch_accuracy",
    "run-validation-tag-epoch_accuracy",
]
extension = ".csv"

legends = [
    "training",
    "validation",
]
for legend, filename in zip(legends, filenames):
    x = []
    y = []
    with open(foldername + filename + extension, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[1]))
            y.append(round(float(row[2]), 4))

    plt.plot(x, y, label=legend)

plt.xlabel('epochs')
plt.ylabel('accuracy')
#plt.title('Interesting Graph\nCheck it out')
plt.legend()
#plt.show()
plt.savefig(foldername + "accuracy", dpi=300)