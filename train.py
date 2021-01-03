import os
import torch
import argparse
import torch.nn as nn
from model import SlowFast
from dataset import VideoDataset
from torch.optim.lr_scheduler import StepLR
from utils import isfloat, print_values, accuracy, checkpoint, load_checkpoint, load_pretrained
from torch.utils.data import DataLoader
from transforms import RandomResizeVideo
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as tv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p',
        '--video_path',
        help='Path to training videos (in JPEG form) (Required)',
        required=True
    )
    parser.add_argument(
        '-v',
        '--validation',
        help='Path to validation videos (in JPEG form) or a percentage of training videos to use between 0.0 and 1.0 (Required)',
        required=True
    )
    parser.add_argument(
        '--model_name',
        help='Name of model (Required)',
        required=True
    )
    parser.add_argument(
        '--model_output',
        help='The output model type, (linear, ...)',
        default='linear'
    )
    parser.add_argument(
        '--loss',
        help='The loss type, (crossentropy, ...)',
        default='crossentropy'
    )
    parser.add_argument(
        '--tensorboard_dir',
        help='Path to tensorboard logging',
        default='./runs/'
    )
    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help='The batch size when training'
    )
    parser.add_argument(
        '--batch_multiplier',
        default=8,
        type=int,
        help='Number of times to accumulate gradients'
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of concurrent DataLoaders'
    )
    parser.add_argument(
        '--model_path',
        default='./model/',
        help='Directory to save model'
    )
    parser.add_argument(
        '--epochs',
        default=120,
        type=int,
        help='Number of times to train over the data'
    )
    parser.add_argument(
        '--input_pixels',
        default=112,
        type=int,
        help='The number of side input size'
    )
    parser.add_argument(
        '--pretrained_path',
        help='Path to pretrained model'
    )
    args = parser.parse_args()
    video_path = args.video_path
    batch_size = args.batch_size
    num_workers = args.num_workers

    valid_is_path = True
    if isfloat(args.validation):
        validation = float(args.validation)
        valid_is_path = False
    else:
        validation = args.validation

    writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.model_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading Dataset')
    train_transforms = Compose([
        RandomResizeVideo((128, 160)) if args.input_pixels == 112 else RandomResizeVideo((256, 320)),
        tv.RandomHorizontalFlipVideo(0.5),
        tv.RandomCropVideo(args.input_pixels),
        tv.NormalizeVideo((128.0, 128.0, 128.0), (128.0, 128.0, 128.0))
    ])
    valid_transforms = Compose([
        RandomResizeVideo((128, 160)) if args.input_pixels == 112 else RandomResizeVideo((256, 320)),
        tv.CenterCropVideo(args.input_pixels),
        tv.NormalizeVideo((128.0, 128.0, 128.0), (128.0, 128.0, 128.0))
    ])

    train_dataset = VideoDataset(video_path, 'train', 64, split=0.0 if valid_is_path else validation,
                                 transform=train_transforms)
    valid_dataset = VideoDataset(validation if valid_is_path else video_path, 'valid', 64,
                                 split=0.0 if valid_is_path else validation, transform=valid_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Train Classes {train_dataset.num_classes()}')
    print(f'Validation Classes {valid_dataset.num_classes()}')

    print('Load Model')
    epoch = 0
    model = SlowFast().to(device)

    if args.model_output == 'linear':
        model_out = nn.Linear(2304, train_dataset.num_classes())
    else:
        raise ValueError('Invalid model_output')

    model_out = model_out.to(device)
    optimizer = torch.optim.SGD(list(model.parameters()) + list(model_out.parameters()), lr=0.005, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Invalid loss')

    if args.pretrained_path:
        model = load_pretrained(args.pretrained_path, model)
    if args.model_path and os.path.exists(args.model_path) and os.path.isdir(args.model_path):
        model, model_out, optimizer, scheduler, epoch = load_checkpoint(
            args.model_path,
            args.model_name,
            args.epochs,
            model,
            model_out,
            optimizer,
            scheduler
        )

    for epoch in tqdm(range(epoch, args.epochs), initial=epoch, total=args.epochs):
        model.train()
        count = 0
        training_loss = 0.0
        multiplier_count = 0
        training_accuracy = 0
        training_5_accuracy = 0
        for inputs, targets in tqdm(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            if multiplier_count == 0:
                optimizer.step()
                optimizer.zero_grad()
                multiplier_count = args.batch_multiplier

            logits = model(inputs)
            if args.model_output == 'linear':
                outputs = model_out(logits)
            else:
                outputs = model_out(logits, targets)

            loss = criterion(outputs, targets) / args.batch_multiplier
            loss.backward()

            multiplier_count -= 1

            training_loss += loss.item() * args.batch_multiplier / targets.size(0)
            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            training_5_accuracy += accuracy(outputs, targets, 5)
            training_accuracy += accuracy(outputs, targets, 1)
            count += targets.shape[0]

        training_accuracy = training_accuracy / count
        training_5_accuracy = training_5_accuracy / count
        training_loss = training_loss / count

        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Accuracy/train', training_accuracy, epoch)
        writer.add_scalar('Top-5 Accuracy/train', training_5_accuracy, epoch)
        print_values('training', training_loss, training_accuracy, training_5_accuracy)

        model.eval()
        count = 0
        validation_loss = 0.0
        validation_accuracy = 0
        validation_5_accuracy = 0
        with torch.no_grad():
            for inputs, targets in tqdm(valid_dataloader):
                inputs = inputs.to(device)
                targets = targets.long().to(device)

                logits = model(inputs)
                if args.model_output == 'linear':
                    outputs = model_out(logits)
                else:
                    outputs = model_out(logits, targets)

                loss = criterion(outputs, targets)

                validation_loss += loss.item() / targets.size(0)
                outputs = outputs.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                validation_5_accuracy += accuracy(outputs, targets, 5)
                validation_accuracy += accuracy(outputs, targets, 1)
                count += targets.shape[0]

        validation_accuracy = validation_accuracy / count
        validation_5_accuracy = validation_5_accuracy / count
        validation_loss = validation_loss / count

        writer.add_scalar('Loss/valid', validation_loss, epoch)
        writer.add_scalar('Accuracy/valid', validation_accuracy, epoch)
        writer.add_scalar('Top-5 Accuracy/valid', validation_5_accuracy, epoch)
        print_values('validation', validation_loss, validation_accuracy, validation_5_accuracy)
        scheduler.step()
        checkpoint(model, model_out, optimizer, scheduler, epoch, args.model_path, args.model_name)


if __name__ == '__main__':
    main()
