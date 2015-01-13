import cPickle
from logistic_sgd import load_data

datasets = load_data('mnist.pkl.gz')

train = datasets[0]
valid = datasets[1]
test  = datasets[2]

grouped_dataset = []
for x in range(10):
    grouped_dataset.append([])
    for y in range(3):
        grouped_dataset[x].append([])
        for z in range(2):
            grouped_dataset[x][y].append([])

for i, s in enumerate([train, valid, test]):
    x = list(s[0].eval())
    y = list(s[1].eval())
    data = zip(x, y)

    for image, classification in data:
        grouped_dataset[classification][i][0].append(image)
        grouped_dataset[classification][i][1].append(classification)

for classification, class_set in enumerate(grouped_dataset):
    for set_type, training_type_set in enumerate(class_set):
        for i in [0,1]:
            grouped_dataset[classification][set_type][i] = np.array(grouped_dataset[classification][set_type][i])

for classification, s in enumerate(grouped_dataset):
    with open('mnist_number_' + str(classification) + '.pkl', 'w') as f:
        cPickle.dump(s, f)