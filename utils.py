from torch.utils.tensorboard import SummaryWriter


def visualise_embeddings(embeddings, labels=None, label_names="Label"):
    print("Embedding")

    writer = SummaryWriter()
    writer.add_embedding(
        mat=embeddings,
        metadata=labels,
        metadata_header=label_names
    )
    print("Embedding done")
