import compute_rouge
from collections import defaultdict

texts = []
summaries = []
hypotheses = []

with open("datasets/test_output.txt") as f:
    for line in f:
        if line.startswith("S"):
            texts.append(line)
        elif line.startswith("T"):
            # print(line)
            summaries.append(" ".join(line.split()[1:]))
            # print(summaries[-1])
        elif line.startswith("H"):
            # print("line for hypo", line)
            hypotheses.append(" ".join(line.split()[2:]))
            # print(hypotheses[-1])

assert len(summaries) == len(hypotheses)

print(compute_rouge.compute_score(hypotheses, summaries))