from sklearn.tree import DecisionTreeClassifier as model
import random
from joblib import dump

text = open("data/pcroomfull.txt", encoding = "utf-8").read()
text = text[0:int(len(text)/5)]

print("loaded")

vocab = ''.join(sorted(set(text)))

char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

def c2i(mystr):
    return([char2int[str(c)] for c in mystr])

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

training = []
labels = []

textgen = model(random_state = 727)

trainingsize = 500

print("splitting")

losp = len(text) - trainingsize

fifty = False

for i in range(len(text) - trainingsize - 1):
    if(fifty and not i/losp > 0.5):
        print("50% done.")
        fifty = True
    training.append(c2i(text[0 + i : trainingsize + i]))
    labels.append(c2i(text[trainingsize + i: trainingsize + i + 1]))

training = np.array(training)
labels = np.array(labels)

print("split. There are %s things to train." % (len(labels)))

maximumsize = -1
training = training[:maximumsize]
labels = labels[:maximumsize]

print("running")

textgen.fit(training, labels.ravel())

print("done")
dump(textgen, "D:/mlpcpcgang2.joblib")
