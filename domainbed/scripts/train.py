# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import socket
import sys
import time
import PIL
import torch.utils.data
import torchvision
import numpy as np

from pathlib import Path

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, FastDataLoader_no_shuffle


def getParameters():
    parser = argparse.ArgumentParser(description='Domain generalization')

    # ------------- Parameters for System -----------------#
    parser.add_argument('--data_dir', type=str, default="Not Setting")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--gpu_id', type=str, default="0", help='visible gpu device id')

    # ------------- Parameters for Task -----------------#
    parser.add_argument('--dataset', type=str, default="CompoundFaults")
    parser.add_argument('--algorithm', type=str, default="AdaClust")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--seen_domains', type=str, nargs='+', default=["HH", "HL", "LH", "LL", "MH", "ML"])
    parser.add_argument('--unseen_domains', type=str, nargs='+',
                        default=["B1H", "B1L", "B2H", "B2L", "B3H", "B3L", "B4H", "B4L"])

    # ------------- HyperParameters -----------------#
    parser.add_argument('--pca_dim', type=int, help='pca dimension')
    parser.add_argument('--offset', type=int, help='start of features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--hparams_json', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--sched', action='store_false', help='Use learning rate scheduler')
    parser.add_argument('--no_pca', action='store_true', help='Clustering without SVD + Truncation step')
    parser.add_argument('--clust_step', type=int, default=None, help='step to perform clustering')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters')

    args = parser.parse_args()

    # ------------- Default Setting -----------------#
    hostname = socket.gethostname()
    if args.data_dir == "Not Setting":
        if hostname == "THINKPAD-X1E":
            args.data_dir = "D:\\datasets\\复合故障数据集\\数据"
        elif hostname == "huangteam-112":
            args.data_dir = "/home/huangteam/wangyuxiang/data/Compound Faults"
        elif hostname == "wang":
            args.data_dir = "E:\\Dataset\\Machinery\\Gear\\复合故障数据集\\数据"
        else:
            raise Exception("Unknown computer host, unable to set data directory")

    if args.output_dir == "Not Setting":
        if hostname == "THINKPAD-X1E":
            args.output_dir = "train_output"
        elif hostname == "huangteam-112":
            args.output_dir = "train_output"
        elif hostname == "wang":
            args.output_dir = "train_output"
        else:
            raise Exception("Unknown computer host, unable to set data directory")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------- Load HyperParameters -----------------#
    from .helpers import get_hparam

    if args.hparams_seed == 0:
        args.hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        args.hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                       misc.seed_hash(args.hparams_seed, args.trial_seed))
    args.hparams = get_hparam(args.hparams, args.hparams_seed)  # To fix hparams for each hparams_seed, else comment out
    if args.pca_dim:
        args.hparams['pca_dim'] = args.pca_dim
    if args.offset:
        args.hparams['offset'] = args.offset
    if args.hparams_json:
        args.hparams.update(json.loads(args.hparams_json))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set flag for AdaClust specific operations
    if "AdaClust" in args.algorithm:
        args.cluster = True
    else:
        args.cluster = False

    # ------------- Print Some Information -----------------#
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    print('HParams:')
    for k, v in sorted(args.hparams.items()):
        print('\t{}: {}'.format(k, v))

    return args


if __name__ == "__main__":
    # ------------- STEP1: Set Default Parameters Before Training -----------------#
    args = getParameters()
    hparams = args.hparams
    device = args.device
    cluster = args.cluster
    data_dir = Path(args.data_dir)
    output_dir = data_dir / (args.dataset + "_" + args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    sys.stdout = misc.Tee(output_dir / 'out.txt')
    sys.stderr = misc.Tee(output_dir / 'err.txt')

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Set number of clusters
    if "AdaClust" in args.algorithm:
        num_clusters = args.num_clusters or hparams["num_clusters"] * dataset.num_classes
        print("NUM CLUSTERS: ", num_clusters)

    # ------------- STEP2: Get Train and Test DataLoader -----------------#
    from .helpers import *
    from .clustering import Faiss_Clustering
    from .preprocess import *

    # Split dataset into training and testing part
    test_data_sep = []
    train_data_sep = []
    eval_loader_names = []
    in_splits = []
    out_splits = []
    train_domain_labels = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(
            env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i)
        )
        if env.name in args.unseen_domains:
            uda, in_ = misc.split_dataset(
                in_, int(len(in_) * args.uda_holdout_fraction), misc.seed_hash(args.trial_seed, env_i)
            )
        if env.name in args.unseen_domains or env.name in args.seen_domains:
            test_data_sep.append(in_)
            eval_loader_names += [env.name + "{}_in".format(env_i)]
            test_data_sep.append(out)
            eval_loader_names += [env.name + "{}_out".format(env_i)]
        if env.name in args.seen_domains:
            train_data_sep.append(in_)
            train_domain_labels.extend([env_i] * len(in_))

    train_data = MyDataloader(train_data_sep)  # Concat train data
    test_data = MyDataloader(test_data_sep)  # Concat test data
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # Generate DataLoaders to Perform Clustering
    num_workers = args.num_workers
    if "AdaClust" in args.algorithm:
        train_loader = FastDataLoader_no_shuffle(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )
        test_loader = FastDataLoader_no_shuffle(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )

    # Generate DataLoaders for DomainBed
    train_idx_split = get_data_split_idx(train_data_sep)
    train_loaders = [
        InfiniteDataLoader(
            dataset=torch.utils.data.Subset(train_data, idx),
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for idx in train_idx_split
    ]
    train_minibatches_iterator = zip(*train_loaders)

    test_idx_split = get_data_split_idx(test_data_sep)
    eval_loaders = [
        FastDataLoader(
            dataset=torch.utils.data.Subset(test_data, idx),
            batch_size=args.batch_size,
            num_workers=num_workers,
        )
        for idx in test_idx_split
    ]

    # ------------- STEP3: Prepare Algorithm and Scheduled Clustering -----------------#
    # Get Algorithm
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(args.seen_domains), hparams)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    algorithm.to(device)

    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = int(len(train_data) / hparams["batch_size"])
    print(f"Number of steps per epoch: {steps_per_epoch}")
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    epochs = int(n_steps / steps_per_epoch)

    if args.sched:
        algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer,
                                                                         T_max=n_steps)  # Initialize scheduler

    # Set clustering schedule
    if cluster:
        cluster_step = args.clust_step
        if args.clust_step is None:
            cluster_step = steps_per_epoch * hparams["clust_epoch"]  # Cluster every hparams["clust_epoch"] epochs
        cluster_step = [(x * cluster_step) for x in range(n_steps) if (x * cluster_step) <= n_steps]
        if hparams["clust_epoch"] == 0:  # cluster every 2**n epochs (0, 1, 2, 4, 8, 16, ...)
            cluster_step = [((2 ** x) * steps_per_epoch) for x in range(epochs) if
                            (2 ** x) <= epochs]  # store the steps at which clustering take place
        print(f"Cluster every {cluster_step} steps")
    else:
        cluster_step = [-1]  # dummy value


    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(args.seen_domains),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, output_dir / filename)


    def return_centroids(algorithm, step):

        with torch.no_grad():
            # Get features and labels # train_features = {ndarray: (N(12573), 2048), 2048 is the length of feature
            # extracted from CNN}
            train_features, train_labels = get_features(train_loader, algorithm, len_train_data, args.batch_size,
                                                        device)
            test_features, test_labels = get_features(test_loader, algorithm, len_test_data, args.batch_size, device)
            train_labels = torch.Tensor(train_labels).type(torch.LongTensor)
            test_labels = torch.Tensor(test_labels).type(torch.LongTensor)

        # Clustering on Train Data
        if args.no_pca:
            train_features2 = train_features  # if no PCA
        else:
            train_features_pca = np.asarray(train_features)
            pca = PCA(hparams["pca_dim"])  # pca_dim = 512 (PCA的长度)
            exp_var = pca.fit(train_features, hparams["offset"])  # offset = 8 (PCAd的起始点)
            train_features2 = pca.apply(
                torch.from_numpy(train_features_pca)).detach().numpy()  # train_features2 = {ndarray: (N, 512)}
            row_sums = np.linalg.norm(train_features2, axis=1)
            train_features2 = train_features2 / row_sums[:, np.newaxis]

        clustering = Faiss_Clustering(train_features2.copy(order="C"), num_clusters)
        clustering.fit()
        cluster_labels_train = get_cluster_labels(clustering, train_features2)  # cluster_labels_train = {list: N}
        images_lists = get_images_list(num_clusters, len_train_data,
                                       cluster_labels_train)  # image_lists = {list: num_clusters}
        train_centroids = torch.empty(
            (len_train_data, train_features.shape[1]))  # train_centroids = {ndarray: (N, 2048)}

        # Get the centroid of the images that share the same cluster in PCA space
        for i, indx in enumerate(images_lists):
            if len(indx) > 0:
                train_centroids[indx] = torch.Tensor(train_features[indx].mean(axis=0))

        # Clustering on Test Data
        if args.no_pca:
            test_features2 = test_features
        else:
            test_features_pca = np.asarray(test_features)
            test_features2 = pca.apply(torch.from_numpy(test_features_pca)).detach().numpy()
            row_sums = np.linalg.norm(test_features2, axis=1)
            test_features2 = test_features2 / row_sums[:, np.newaxis]

        cluster_labels_test = get_cluster_labels(clustering, test_features2)
        images_lists = get_images_list(num_clusters, len_test_data, cluster_labels_test)
        test_centroids = torch.empty((len_test_data, test_features.shape[1]))

        for i, indx in enumerate(images_lists):
            if len(indx) > 0:
                test_centroids[indx] = torch.Tensor(test_features[indx].mean(axis=0))

        return train_centroids, test_centroids, exp_var


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        epoch = step / steps_per_epoch

        if step == 0 or (step in cluster_step):
            if cluster:
                train_centroids, test_centroids, exp_var = return_centroids(algorithm, step)
                print("calculate centroids successfully!")
            else:
                test_centroids = None

        if cluster:
            minibatches_device = [
                (x.to(device), train_centroids[idx].to(device), y.to(device))
                for ((x, y), idx) in next(train_minibatches_iterator)
            ]
        else:
            minibatches_device = [
                (x.to(device), y.to(device)) for ((x, y), idx) in next(train_minibatches_iterator)
            ]

        step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            if cluster:
                results["exp_var"] = str(exp_var)

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders)
            for i, (name, loader) in enumerate(evals):
                acc, detail_acc = misc.accuracy(algorithm, loader, None, device, dataset.num_classes, test_centroids)
                results[name + '_acc'] = acc
                for acc_item, idx in enumerate(detail_acc):
                    results[name + '_acc' + str(acc_item) + '_acc'] = idx

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=24)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = output_dir / 'results.json'
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(output_dir / 'done', 'w') as f:
        f.write('done')
