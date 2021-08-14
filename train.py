from tools.model import *
from tools.nltk_utils import *

with open('data/intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!', ',', "\'"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
lr = 0.001
input_size = len(X_train[0])
hidden_size = 8
num_classes = len(tags)
print(input_size, num_classes)

model = ChatBotModule(input_size, hidden_size, num_classes, lr)

data = ChatbotDataModule(ChatDataset(X_train, y_train), batch_size=batch_size)

trainer = pl.Trainer(max_epochs=num_epochs, default_root_dir="model/")
trainer.fit(model, data)
trainer.save_checkpoint("data/model.ckpt")


data = {
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": num_classes,
"all_words": all_words,
"tags": tags
}

FILE = "data/params.pth"
torch.save(data, FILE)

print(f'training complete. file saved to "data/model.ckpt" and "{FILE}"')