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
            rec_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss += rec_loss.item()
            batches_log.append(rec_loss.item())
        losses_log.append(iter_loss)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.4f}")

    return losses_log, batches_log
