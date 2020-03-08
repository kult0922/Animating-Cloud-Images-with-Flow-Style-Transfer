from data.video_folder import VideoFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

def make_dataloader(opt):

    trainset = VideoFolder(root=os.path.join(opt.dataroot, 'train'),
                           nframes = opt.nframes,
                           transform=transforms.Compose([
                           transforms.Resize( (opt.imageSize, opt.imageSize) ),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                           )

    testset = VideoFolder( root=os.path.join(opt.dataroot, 'test'),
                           nframes = opt.nframes,
                           transform=transforms.Compose([
                           transforms.Resize((opt.imageSize, opt.imageSize)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                         )


    print('trainset size ' + str(len(trainset)))
    print('testset size ' + str(len(testset)))

    train_loader = DataLoader(trainset,
                          batch_size=opt.batchSize,
                          num_workers=int(opt.workers),
                          shuffle=not opt.noShuffle,
                          drop_last = True,
                          pin_memory=True
                          )

    valid_loader = DataLoader(testset,
                          batch_size=opt.batchSize,
                          num_workers=1,
                          shuffle=not opt.noShuffle,
                          drop_last = True,
                          pin_memory=False
                          )

    print('trainloader size: ' + str(len(train_loader)))
    print('validloader size: ' + str(len(valid_loader)))

    return train_loader, valid_loader

def make_demo_dataloader(opt):
    demoset = VideoFolder( root=os.path.join(opt.dataroot, 'test'),
                           nframes = opt.nframes,
                           transform=transforms.Compose([
                           transforms.Resize((opt.imageSize, opt.imageSize)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                           demo=True
                         )
    print('demoset size ' + str(len(demoset)))

    demo_loader = DataLoader(demoset,
                          batch_size=opt.batchSize,
                          num_workers=1,
                          shuffle=not opt.noShuffle,
                          drop_last = True,
                          pin_memory=False
                          )

    print('demoloader size: ' + str(len(demo_loader)))

    return demo_loader

def make_test_dataloader(opt):
    testset = VideoFolder( root=opt.drivingVideo,
                           nframes = opt.nframes,
                           transform=transforms.Compose([
                           transforms.Resize((opt.imageSize, opt.imageSize)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                           test=True
                         )

    test_loader = DataLoader(testset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=not opt.noShuffle,
                          drop_last = True,
                          pin_memory=False
                          )

    return test_loader
