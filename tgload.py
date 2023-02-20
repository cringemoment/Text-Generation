from joblib import load

text = open("data/welcometopcgang.txt", encoding = "utf-8").read()

vocab = ''.join(sorted(set(text)))

char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def fillinrest(mstring, objective):
    temp = " " * (objective - len(mstring))
    return(temp + mstring)

print("loading")
textgenerator = load("D:/dtcpcgang1.joblib")
testpredict = "w"
print("loaded")

features = 20

testpredict = fillinrest(testpredict, features)
inttestpredict = [char2int[char] for char in testpredict]

while True:
    cinttestpredict = [inttestpredict[len(inttestpredict) - features: len(inttestpredict)]]
    output = textgenerator.predict(cinttestpredict)
    inttestpredict.append(output[0][0])
    if(len(inttestpredict) > 1000):
        break

testpredict = flatten_list([flatten_list(flatten_list(int2char[char] for char in inttestpredict))])
print(''.join(testpredict))
