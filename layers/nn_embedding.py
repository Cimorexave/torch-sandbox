import torch
print("\n"*5)

vocab_size = 10
embedding_dim = 3
embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
input_indices = torch.tensor([1,5,0,8])
embedded_vectors = embedding_layer(input_indices)
print(f"Input indices: {input_indices}")
print(f"Embedded vectors:\n{embedded_vectors}")

# created vector for each input index, each vector has dimension of embedding_dim (3 in this case)