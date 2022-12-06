from torch.utils.tensorboard import SummaryWriter
from time import time


def visualise_embeddings(embeddings, labels=None, label_names="Label"):
    print("Embedding")

    writer = SummaryWriter()
    start = time()
    writer.add_embedding(
        mat=embeddings,
        metadata=labels,
        metadata_header=label_names
    )
    print(f"Total time:", time() - start)

    print("Embedding done")
