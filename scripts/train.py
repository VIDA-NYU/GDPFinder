from models import target_distribution


def train_reconstruction(
    model, loader, loss, optimizer, device, epochs=100, verbose=True
):
    losses_log = []
    batches_log = []
    for i in range(epochs):
        iter_loss = 0
        for batch, _ in loader:
            batch = batch.to(device)
            _, decoded = model(batch)
            rec_loss = loss(decoded, batch)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            iter_loss += rec_loss.item() / batch.shape[0]
            batches_log.append(rec_loss.item())
        losses_log.append(iter_loss)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

    return losses_log, batches_log


def train_clustering(model, loader, loss, optimizer, device, epochs=100, verbose=True):
    losses_log = []
    batches_log = []

    for i in range(epochs):
        iter_loss = 0
        for batch, _ in loader:
            batch = batch.to(device)
            output = model(batch)
            target = target_distribution(output).detach()
            cluster_loss = loss(output.log(), target) / output.shape[0]
            optimizer.zero_grad()
            cluster_loss.backward()
            optimizer.step()
            iter_loss += cluster_loss.item()
            batches_log.append(cluster_loss.item())
        losses_log.append(iter_loss)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

    return losses_log, batches_log
