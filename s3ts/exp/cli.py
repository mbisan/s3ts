#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Command Line Interface (CLI) for the synthetic experiments. """

# package imports
from s3ts.legacy.training import run_model, save_results
from s3ts.data.acquisition import download_dataset
from s3ts.experiments.setup import setup_pretrain_dm
from s3ts.experiments.setup import setup_train_dm 
from s3ts.experiments.setup import train_test_splits

# standard library
from pathlib import Path
import logging as log
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    # compulsory parameters
    parser.add_argument('--dset', type=str, required=True,
        help='Name of the dataset from which create the DTWs')
    parser.add_argument('--dsrc', type=str, required=True, choices=['ts', 'df', 'gf'],
        help=r'Data source for the model {ts: time series, df: dissimilairty frames, gf: gramian frames}')
    parser.add_argument('--arch', type=str, required=True, choices=['nn', 'rnn', 'cnn', 'res'],
        help='Name of the architecture from which create the model')
    parser.add_argument('--use_pretrain', type=bool, action=argparse.BooleanOptionalAction, 
                        default=False, help='Use pretrained encoder or not (df/gf mode only)')
    parser.add_argument('--pretrain_mode', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Switch between train and pretrain mode')
    parser.add_argument('--rho_dfs', type=float, default=0.1, help='Forgetting parameter (DF only)')

    # window parameters
    parser.add_argument('--wlen', type=int, default=10, help='Window length')
    parser.add_argument('--wdw_str', type=int, default=1, help='Window stride')
    parser.add_argument('--str_str', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Whether to stride the stream during pretrain')

    #
    parser.add_argument('--enc_feats', type=int, default=None, help='Encoder complexity hyperparameter.')
    parser.add_argument('--dec_feats', type=int, default=64, help='Decoder complexity hyperparameter.')
    
    parser.add_argument('--exc', type=int, default=16,
                        help='Number of events per class')

    parser.add_argument('--train_strat_size', type=int, default=2,
                        help='Stratification size for the training STS')

    parser.add_argument('--train_event_mult', type=int, default=4,
                        help='Event multiplier for the training STS')
    
    parser.add_argument('--train_exc_limit', type=int, default=None,
                        help='Limit the number of available events per class (experimentation purposes)')

    parser.add_argument('--test_sts_length', type=int, default=200,
                        help='Length of the STS used for testing')
    
    parser.add_argument('--pret_sts_length', type=int, default=1000,
                        help='Length of the STS used for pretraining')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')
    
    parser.add_argument('--val_size', type=float, default=0.25,
                        help='Validation set size (as a fraction of the training set)')
    
    parser.add_argument('--max_epochs', type=int, default=60,
                        help='Maximum number of epochs for the training')
    
    parser.add_argument('--learning_rate', type=float, default=1E-4,
                        help='Value of the learning rate')
    
    parser.add_argument('--cv_rep', type=int, default=0,
                        help='Cross-validation repetition number')

    parser.add_argument('--res_fname', type=str, default="results.csv",
                        help='Results file for the training')
    
    parser.add_argument('--train_dir', type=str, default="training/",
                        help='Directory for the training files')

    args = parser.parse_args()
   
    # ~~~~~~~~~~~~ Put the arguments in variables ~~~~~~~~~~~~
    
    # Basic Information
    dataset: str = args.dataset
    mode: str = args.mode
    arch: str = args.arch
    use_pretrain: bool = args.use_pretrain
    pretrain_mode: bool = args.pretrain_mode
    
    # Model parameters
    rho_dfs: float = args.rho_dfs
    window_length: int = args.window_length
    stride_series: bool = args.stride_series
    window_time_stride: int = args.window_time_stride
    window_patt_stride: int = args.window_patt_stride
    num_encoder_feats: int = args.num_encoder_feats
    num_decoder_feats: int = args.num_decoder_feats

    # Training parameters
    exc: int = args.exc
    train_event_mult: int = args.train_event_mult
    train_exc_limit: int = args.train_exc_limit
    train_strat_size: int = args.train_strat_size
    test_sts_length: int = args.test_sts_length
    pret_sts_length: int = args.pret_sts_length
    batch_size: int = args.batch_size
    val_size: float = args.val_size
    max_epochs: int = args.max_epochs
    learning_rate: float = args.learning_rate
    random_state: int = args.random_state
    cv_rep: int = args.cv_rep

    # Paths
    log_file: Path = Path(args.log_file)
    res_fname: Path = Path(args.res_fname)
    train_dir: Path = Path(args.train_dir)
    storage_dir: Path = Path(args.storage_dir)
    num_workers: int = args.num_workers

    # Print the arguments in a nice way for debugging
    log.info("Input Parameters:")
    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")

    # ~~~~~~~~~~~~ Launch the training loop ~~~~~~~~~~~~
    main_loop(
        dataset=dataset,
        mode=mode,
        arch=arch,
        use_pretrain=use_pretrain,
        pretrain_mode=pretrain_mode,
        rho_dfs=rho_dfs,
        window_length=window_length,
        stride_series=stride_series,
        window_time_stride=window_time_stride,
        window_patt_stride=window_patt_stride,
        num_encoder_feats=num_encoder_feats,
        num_decoder_feats=num_decoder_feats,
        exc=exc,
        train_event_mult=train_event_mult,
        train_strat_size=train_strat_size,
        train_exc_limit=train_exc_limit,
        test_sts_length=test_sts_length,
        pret_sts_length=pret_sts_length,
        batch_size=batch_size,
        val_size=val_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        cv_rep=cv_rep,
        log_file=log_file,
        res_fname=res_fname,
        train_dir=train_dir,
        storage_dir=storage_dir,
        num_workers=num_workers)
