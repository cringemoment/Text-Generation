import random
from collections import defaultdict

# Step 1: Read in the text file
with open("data\jaykobgeneral03252023.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("loaded")

# Step 2: Preprocess the text
tokens = [char for char in text]
print("split")

# Step 3: Build the transition matrix
order = 8
transition_matrix = defaultdict(lambda: defaultdict(int))

print("starting matrix")
for i in range(len(tokens) - order):
    current_state = tuple(tokens[i:i+order])
    next_state = tokens[i+order]
    transition_matrix[current_state][next_state] += 1
print("for loop 1 done")
for state, next_state_counts in transition_matrix.items():
    total_count = sum(next_state_counts.values())
    for next_state in next_state_counts:
        next_state_counts[next_state] /= total_count
print("done matrix")

# Step 4: Generate new text
length = 1000
seed = random.choice(list(transition_matrix.keys()))

generated_text = list(seed)

for i in range(length):
    current_state = tuple(generated_text[-order:])
    next_token = random.choices(list(transition_matrix[current_state].keys()), list(transition_matrix[current_state].values()))[0]
    generated_text.append(next_token)

generated_text = "".join(generated_text)

print(generated_text)
