import torch
from torchvision.models import resnet50
from tqdm import tqdm

import colossalai
from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR


def synthesize_data():
    img = torch.rand(gpc.config.BATCH_SIZE, 3, 32, 32)
    label = torch.randint(low=0, high=10, size=(gpc.config.BATCH_SIZE,))
    return img, label


def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # trace the model with meta data
    model = resnet50(num_classes=10).cuda()
    input_sample = {'x': torch.rand([gpc.config.BATCH_SIZE * torch.distributed.get_world_size(), 3, 32, 32]).to('meta')}

    model = autoparallelize(model, input_sample)
    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    for epoch in range(gpc.config.NUM_EPOCHS):
        model.train()

        # if we use synthetic data
        # we assume it only has 10 steps per epoch
        num_steps = range(10)
        progress = tqdm(num_steps)

        for _ in progress:
            # generate fake data
            img, label = synthesize_data()

            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            train_loss = criterion(output, label)
            train_loss.backward(train_loss)
            optimizer.step()
        lr_scheduler.step()

        # run evaluation
        model.eval()
        correct = 0
        total = 0

        # if we use synthetic data
        # we assume it only has 10 steps for evaluation
        num_steps = range(10)
        progress = tqdm(num_steps)

        for _ in progress:
            # generate fake data
            img, label = synthesize_data()

            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = model(img)
                test_loss = criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
            ranks=[0])


if __name__ == '__main__':
    main()
#
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0.01)
    logger.info("Start training")
    for step, (img, label) in enumerate(tqdm(train_dataloader)):
        img, label = img.to(gpc.main_device), label.to(gpc.main_device)
        optimizer.zero_grad()
        logits = model(img)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
    logger.info("Start testing")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img, label = img.to(gpc.main_device), label.to(gpc.main_device)
            logits = model(img)
            loss = F.cross_entropy(logits, label, reduction='sum')
            test_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    accuracy = correct / len(test_dataloader.dataset)
    logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)}, {accuracy:.4f}")
