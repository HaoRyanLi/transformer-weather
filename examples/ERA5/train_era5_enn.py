import tensorflow as tf
tf.config.list_physical_devices('GPU')

import sys
sys.path.insert(0, "/work/09012/haoli1/ls6/transformer-weather")
import logging
import torch
from torch.optim.lr_scheduler import ExponentialLR

from trphysx.config.configuration_auto import AutoPhysConfig
from trphysx.embedding.embedding_auto import AutoEmbeddingModel
from trphysx.viz.viz_auto import AutoViz
from trphysx.embedding.training import *
import argparse
import gdown
import os


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    directory = '/scratch/09012/haoli1/ERA5/dataset/'
    train_data_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        train_data_list.append(f)
    train_data_files = ','.join(train_data_list)
    valid_data_files = '/scratch/09012/haoli1/ERA5/val_dataset/era5_train_09012020_3_24hr.npz'

    sys.argv = sys.argv + ["--exp_name", "era5"]
    sys.argv = sys.argv + ["--training_h5_file", train_data_files]
    sys.argv = sys.argv + ["--eval_h5_file", valid_data_files]
    sys.argv = sys.argv + ["--batch_size", "64"]
    sys.argv = sys.argv + ["--block_size", "24"]
    sys.argv = sys.argv + ["--stride", "24"]
    sys.argv = sys.argv + ["--n_train", "27"]
    sys.argv = sys.argv + ["--n_eval", "6"]
    sys.argv = sys.argv + ["--epochs", "10"]
    sys.argv = sys.argv + ["--n_gpu", "3"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    args = EmbeddingParser().parse()

    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: {}".format(args.device))
    logger.info("epochs: {}".format(args.epochs))
    
    # Load transformer config file
    config = AutoPhysConfig.load_config(args.exp_name)

    data_handler = AutoDataHandler.load_data_handler(args.exp_name)
    viz = AutoViz.load_viz(args.exp_name, plot_dir=args.plot_dir)

     # Set up data-loaders
    training_loader = data_handler.createTrainingLoader(
                        args.training_h5_file, 
                        block_size=args.block_size, 
                        stride=args.stride, 
                        ndata=args.n_train, 
                        batch_size=args.batch_size)
    testing_loader = data_handler.createTestingLoader(
                        args.eval_h5_file, 
                        block_size=32, 
                        ndata=args.n_eval, 
                        batch_size=8)

    # Set up model
    model = AutoEmbeddingModel.init_trainer(args.exp_name, config).to(args.device)
    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(args.device)
    model.embedding_model.std = std.to(args.device)
    if args.epoch_start > 1:
        model.load_model(args.ckpt_dir, args.epoch_start)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.995**(args.epoch_start-1), weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler), viz)
    trainer.train(training_loader, testing_loader)
    model.embedding_model.save_model('/work/09012/haoli1/ls6/transformer-physx/examples/cylinder', epoch=args.epochs)
