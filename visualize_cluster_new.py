from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
import os
import torch
import pickle
import numpy as np
directory = "bert_embeddings_500_base_uncased/abstract_embedding"
folders =  os.listdir(directory)
doc_ids = []
data = []

with open("bert_embeddings_500_base_uncased_abstract_doc_info.pkl", "rb") as input_file:
  p1 = pickle.load(input_file)

for folder in folders:
  path = directory + "/" + folder + "/pdf_json"
  files = os.listdir(path)
  print(path, len(files))
  for f in files:
    doc_ids.append(f)
    embedding = torch.load(path+"/"+f)
    embedding = embedding.reshape(768)
    data.append(list(embedding.data.cpu().numpy()))
print("Loaded data")
print(len(data))

X_2d = tsne.fit_transform(data)
new_arr = []
for i in range(len(doc_ids)):
  new_arr.append(p1[doc_ids[i]])

n__ = np.asarray(new_arr)
X_fin = np.vstack([X_2d, n__]).T

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
color = ['r', 'k', 'c', 'g', 'b', 'm']
j = 0
for i in range(5):
  plt.scatter(X_fin[:, 0, y==i], X_fin[:, 1, y==i], c=color[j])
  j += 1
plt.savefig('abstract_cluster.png')

plt.show()
