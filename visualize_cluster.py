from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
import os
import torch
directory = "bert_embeddings_500_base_uncased/document_embedding"
folders =  os.listdir(directory)
doc_ids = []
data = []

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

print(X_2d.shape)
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c='r')
plt.savefig('document_cluster.png')

plt.show()
