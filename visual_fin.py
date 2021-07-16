from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=0)
#tsne1 = TSNE(n_components=2, random_state=0)
import os
import torch
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

directory = "new_embeddings/bio_cased/abstract_embedding_cluster_id_to_doc_ids_19.pkl"
with open(directory, 'rb') as f:
  temp = pickle.load(f)

dir_ = "new_embeddings/bio_cased/abstract_embedding"


ax = plt.axes(projection='3d')
color = ['r', 'b', 'g', 'c', 'm', 'y', 'k', '#9c9fdb']
j = 0
for key,value in temp.items():
  data = []
  files = os.listdir(dir_)
  for f in files:
    if f in value:
      embedding = torch.load(dir_+"/"+f)
      embedding = embedding.reshape(768)
      data.append(list(embedding.data.cpu().numpy()))
  X_2d = tsne.fit_transform(data)

  ax.scatter(X_2d[:, 0], X_2d[:, 1], X_2d[:, 2], c=color[j])
  #plt.scatter(X_2d[:, 2], X_2d[:, 1], c=color[j])
  j+=1    

plt.savefig('abstract_bio_cluster.png')
