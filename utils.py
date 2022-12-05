from torch.utils.tensorboard import SummaryWriter


def visualise_embeddings(embeddings, labels=None):
    print("Embedding")

    writer = SummaryWriter()
    writer.add_embedding(
        mat=embeddings,
        metadata=labels
    )
    print("Embedding done")
