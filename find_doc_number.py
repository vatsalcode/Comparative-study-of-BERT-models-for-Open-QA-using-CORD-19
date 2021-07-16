import pickle

for i in range(20):
  with open('new_embeddings_mean/sci_uncased/document_embedding_cluster_id_to_doc_ids_{}.pkl'.format(i), 'rb') as f:
    data = pickle.load(f)

  print('Number of clusters', len(data))

  for key, value in data.items():
    print('Cluster {} : {} documents'.format(key, len(value)))

