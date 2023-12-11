from helper_functions import *
import multiprocessing

from s3ts.api.nets.methods import create_model_from_DM, train_model

from argparse import ArgumentParser

def main(args):

    dm = load_dmdataset(
        args.dataset, dataset_home_directory=args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers, 
        window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size, 
        compute_n=args.compute_n, subjects_for_test=args.subjects_for_test)

    model = create_model_from_DM(dm, name=None, 
        dsrc="img", arch=args.encoder_architecture, task="cls", lr=args.lr)
    
    model, data = train_model(dm, model, max_epochs=args.max_epochs)
    print(data)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str,
        help="Dataset name for training")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, 
        help="Directory of the dataset")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count()//2, type=int)
    parser.add_argument("--window_size", default=32, type=int, 
        help="Window size of the dissimilarity frames fed to the classifier")
    parser.add_argument("--window_stride", default=1, type=int, 
        help="Stride used when extracting windows")
    parser.add_argument("--normalize", action="store_true", 
        help="Wether to normalize the dissimilarity frames and STS")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--pattern_size", default=32, type=int, 
        help="Size of the pattern for computation of dissimilarity frames (not used)")
    parser.add_argument("--compute_n", default=500, type=int, 
        help="Number of samples extracted from the STS or Dissimilarity frames to compute medoids and/or means for normalization")
    parser.add_argument("--subjects_for_test", nargs="+", type=int, 
        help="Subjects reserved for testing and validation")
    parser.add_argument("--encoder_architecture", default="cnn", type=str, 
        help="Architecture used for the encoder")
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")