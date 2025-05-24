import configargparse
import logging
import random
import os
import sys
import socket
import numpy as np
import torch
import dask.array as da
import pandas as pd
# import seaborn as sns # Already present in the original file
# import matplotlib.pyplot as plt # Already present in the original file

from pathlib import Path
from typing import List, Tuple, Union

from .snp_reader import SNPReader
from ..model.switchers import Switchers
# from ..model.initializations import load_or_compute_pca # This will be imported in train.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_train_args(argv: List[str]):
    """Training arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture train',
                                           description='Rapid population clustering with autoencoders - training mode',
                                           config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           allow_abbrev=False) # Prevent abbreviation to avoid conflicts
    
    # Existing arguments
    parser.add_argument('--epochs', required=False, type=int, default=250, help='Maximum number of epochs.')
    parser.add_argument('--batch_size', required=False, default=800, type=int, help='Batch size.')
    parser.add_argument('--learning_rate', required=False, default=25e-4, type=float, help='Learning rate.')

    parser.add_argument('--initialization', required=False, type=str, default = 'gmm',
                        choices=['kmeans', 'gmm', 'supervised', 'random'], help='P initialization method.')
    
    parser.add_argument('--activation', required=False, default='relu', type=str, choices=['relu', 'tanh', 'gelu'], help='Activation function for encoder layers.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed for reproducibility.')
    
    # Modified --k argument
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters. Not required if --auto_k_dbscan is used, but can serve as a fallback if DBSCAN fails to find clusters.')
    
    parser.add_argument('--hidden_size', required=False, default=256, type=int, help='Dimension of the hidden layer in the encoder.')
    parser.add_argument('--pca_path', required=False, type=str, help='Path to store/load PCA object. If not provided, it will be constructed relative to save_dir (e.g., save_dir/name_pca.pt).')
    parser.add_argument('--pca_components', required=False, type=int, default=8, help='Number of principal components to use for PCA-based initializations and DBSCAN if auto_k_dbscan is enabled.')
    parser.add_argument('--save_dir', required=True, type=str, help='Directory to save model outputs, logs, and PCA object (if pca_path is not specified).')
    parser.add_argument('--data_path', required=True, type=str, help='Path to the input genotype data file (e.g., VCF, BED, PGEN, HDF5, NPY).')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name, used for naming output files.')
    parser.add_argument('--imputation', type=str, default='mean', choices=['mean', 'zero'], help='Imputation method for missing genotype data (default: mean).')
    
    parser.add_argument('--supervised_loss_weight', required=False, default=100, type=float, help='Weight for the supervised classification loss component if --supervised is used.')
    parser.add_argument('--populations_path', required=False, default='', type=str, help='Path to a file containing population labels for supervised mode. Each line corresponds to a sample.')
    parser.add_argument('--supervised', action='store_true', default=False, help='Enable supervised mode. Requires --populations_path.')
    
    parser.add_argument('--num_gpus', required=False, default=0, type=int, help='Number of GPUs to use for training. Set to 0 for CPU-only.')
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPU cores to use (e.g., for DBSCAN n_jobs, DataLoader num_workers).')

    # New DBSCAN arguments
    parser.add_argument('--auto_k_dbscan', action='store_true', default=False, help='Automatically determine K using DBSCAN on PCA components instead of using a fixed K.')
    parser.add_argument('--dbscan_eps', required=False, type=float, default=0.5, help='DBSCAN eps parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Used if --auto_k_dbscan is enabled.')
    parser.add_argument('--dbscan_min_samples', required=False, type=int, default=5, help='DBSCAN min_samples parameter: The number of samples in a neighborhood for a point to be considered as a core point. Used if --auto_k_dbscan is enabled.')
    
    args = parser.parse_args(argv)

    # Post-parsing validation
    if args.auto_k_dbscan and args.k is not None:
        log.info("    Both --auto_k_dbscan and --k are specified. --k will be used as a fallback if DBSCAN fails.")
    elif not args.auto_k_dbscan and args.k is None:
        parser.error("--k is required if --auto_k_dbscan is not used.")
    if args.supervised and not args.populations_path:
        parser.error("--supervised mode requires --populations_path to be specified.")

    return args

def parse_infer_args(argv: List[str]):
    """Inference arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture infer',
                                     description='Rapid population clustering with autoencoders - inference mode',
                                     config_file_parser_class=configargparse.YAMLConfigFileParser,
                                     allow_abbrev=False)
    parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode.')
    parser.add_argument('--save_dir', required=True, type=str, help='Load model from this directory.')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data.')
    parser.add_argument('--name', required=True, type=str, help='Trained experiment/model name.')
    parser.add_argument('--batch_size', required=False, default=1000, type=int, help='Batch size.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPUs to be used in the execution.')

    return parser.parse_args(argv)

def read_data(tr_file: str, master: bool, tr_pops_f: str=None, imputation: str='mean') -> Tuple[da.core.Array, Union[List[str], None]]:
    """
    Reads SNP data from a file and applies imputation if specified.
    Also reads population labels if provided.

    Args:
        tr_file (str): Path to the SNP data file.
        master (bool): Whether this process is the master for printing output.
        tr_pops_f (str, optional): Path to the file containing population labels. Defaults to None.
        imputation (str): Type of imputation to apply ('mean' or 'zero'). Defaults to 'mean'.


    Returns:
        Tuple[da.core.Array, Union[List[str], None]]: A Dask array containing the SNP data,
                                                      and a list of population strings or None.
    """
    snp_reader = SNPReader()
    data = snp_reader.read_data(tr_file, imputation, master)
    if master:
        log.info(f"    Data contains {data.shape[0]} samples and {data.shape[1]} SNPs.")
    
    tr_pops_list = None
    if tr_pops_f:
        try:
            with open(tr_pops_f, 'r') as fb:
                tr_pops_list = [p.strip() for p in fb.readlines()]
            if master:
                log.info(f"    Read {len(tr_pops_list)} population labels from {tr_pops_f}.")
            if len(tr_pops_list) != data.shape[0]:
                log.error(f"    Mismatch between number of samples in data ({data.shape[0]}) and population labels ({len(tr_pops_list)}).")
                sys.exit(1)
        except FileNotFoundError:
            log.error(f"    Populations file not found: {tr_pops_f}")
            sys.exit(1)
    
    return data, tr_pops_list

def initialize_data(master: bool, trX_dask: da.core.Array, tr_pops_list: Union[List[str], None]=None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Initialize data by computing Dask array to NumPy array.
    Converts population list to NumPy array if present.

    Args:
        master (bool): Whether this process is the master for printing output.
        trX_dask (da.core.Array): The training data as a Dask array.
        tr_pops_list (Union[List[str], None]): List of population labels or None.

    Returns:
        Tuple[np.ndarray, Union[np.ndarray, None]]: The initialized training data as a NumPy array,
                                                     and population labels as a NumPy array or None.
    """
    if master:
        log.info("    Bringing data into memory (computing Dask array)...")
    
    data_np = trX_dask.compute() # This can be memory intensive for large datasets

    if master:
        log.info(f"    Data successfully loaded into memory. Shape: {data_np.shape}")
        log.info("")

    y_np = None
    if tr_pops_list is not None:
        y_np = np.array(tr_pops_list)
        
    return data_np, y_np

def train(initialization_method: str, device: torch.device, save_dir : str, name: str, 
        k_val: int, seed: int, pca_n_components: int, epochs: int, batch_size: int, learning_rate: float, 
        data_np: np.ndarray, num_gpus: int, activation_str: str, hidden_size: int, master: bool, num_cpus: int,
        y_np: Union[np.ndarray, None], supervised_loss_weight: float) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    """
    Train the model using specified initialization, hyperparameters, and data.
    This function now expects data_np (NumPy array) and y_np (NumPy array or None).

    Args:
        initialization_method (str): Initialization strategy to use.
        device (torch.device): Device to perform training (CPU or GPU).
        save_dir (str): Directory to save model initialization files.
        name (str): Name identifier for the training run.
        k_val (int): Number of clusters or components (the determined K).
        seed (int): Random seed for reproducibility.
        pca_n_components (int): Number of PCA components for the chosen initialization method's internal PCA.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        data_np (np.ndarray): Training data as a NumPy array.
        num_gpus (int): Number of GPUs to use for DDP.
        activation_str (str): String key for selecting the activation function.
        hidden_size (int): Size of hidden layers in the neural network.
        master (bool): Whether this process is the master for printing output.
        num_cpus (int): Number of CPUs for data loading.
        y_np (Union[np.ndarray, None]): Population labels as a NumPy array or None.
        supervised_loss_weight (float): Weight for supervised loss.


    Returns:
        Tuple[np.ndarray, np.ndarray, torch.nn.Module]: Trained P and Q matrices (as NumPy arrays)
                                                         and the raw trained model instance.
    """
    # Determine PCA object path for the initialization method (might be different from DBSCAN's PCA)
    # The initialization methods themselves handle PCA loading/computation internally.
    # The `pca_path` argument in `initializations.get_decoder_init` is constructed like this:
    init_pca_obj_filename = f'{name}_pca.pt' # This is the standard name used by initializations.py
    init_pca_obj_path = str(Path(save_dir) / init_pca_obj_filename)
    
    switchers = Switchers.get_switchers()
    activation_fn = switchers['activations'][activation_str](0) # Create activation instance

    # Call the selected initialization function
    # Note: `data_np` is passed here. `y_np` is also passed for supervised initialization.
    P_np, Q_np, raw_model_instance = switchers['initializations'][initialization_method](
        epochs, batch_size, learning_rate, k_val, seed, 
        init_pca_obj_path, # Path for the initialization's PCA object
        name, pca_n_components, # pca_n_components for the initialization
        data_np, # NumPy data
        device, 
        num_gpus, hidden_size, activation_fn, master, num_cpus, 
        y_np, # NumPy population labels
        supervised_loss_weight
    )
    
    return P_np, Q_np, raw_model_instance

def write_outputs(Q: np.ndarray, run_name: str, K_val: int, out_path_str: str, P: np.ndarray=None) -> None:
    """
    Save the Q and optional P matrices to specified output files.
    K_val is the actual K value used for training.

    Args:
        Q (numpy.ndarray): Q matrix to be saved.
        run_name (str): Identifier for the run, used in file naming.
        K_val (int): Number of clusters used, included in the file name.
        out_path_str (str): Directory path where the output files should be saved.
        P (numpy.ndarray, optional): P matrix to be saved, if provided. Defaults to None.

    Returns:
        None
    """
    out_path = Path(out_path_str)
    out_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    
    q_filename = out_path / f"{run_name}.{K_val}.Q"
    np.savetxt(q_filename, Q, delimiter=' ')
    log.info(f"    Q matrix saved to: {q_filename}")

    if P is not None:
        p_filename = out_path / f"{run_name}.{K_val}.P"
        np.savetxt(p_filename, P, delimiter=' ')
        log.info(f"    P matrix saved to: {p_filename}")
    return 

def find_free_port(start_port=12355):
    """ Find a free port starting from a given port number. """
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port # Port is free
        except OSError: # Port is already in use
            port += 1
        if port > 65535:
            raise RuntimeError("Could not find a free port.")


def ddp_setup(stage: str, rank: int, world_size: int) -> None:
    """
    Initialize or destroy the Distributed Data Parallel (DDP) process group.

    Args:
        stage (str): 'begin' initializes, any other value destroys.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes in the DDP group.

    Returns:,
        None
    """
    if world_size > 1: # Only setup DDP if more than one GPU/process
        if stage == 'begin':
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port()) # Ensure a free port is found
            if torch.cuda.is_available():
                torch.cuda.set_device(rank) # Set device for this DDP process
                backend = "nccl"
            else: # Fallback for CPU DDP (less common, usually for testing)
                backend = "gloo"
            
            log.debug(f"    DDP Setup: Initializing process group. Rank: {rank}, World Size: {world_size}, Backend: {backend}, Addr: {os.environ['MASTER_ADDR']}, Port: {os.environ['MASTER_PORT']}")
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
            log.debug(f"    DDP Setup: Process group initialized for rank {rank}.")
        else:
            if torch.distributed.is_initialized():
                log.debug(f"    DDP Setup: Destroying process group for rank {rank}.")
                torch.distributed.destroy_process_group()
                log.debug(f"    DDP Setup: Process group destroyed for rank {rank}.")
            else:
                log.debug(f"    DDP Setup: Attempted to destroy uninitialized process group for rank {rank}. Skipping.")


def process_cv_loglikelihood(cv_loglikelihood: list) -> pd.DataFrame:
    """
    Process cross-validation errors and return a reduced DataFrame with mean and standard deviation.

    Args:
        cv_loglikelihood (list): List of cross-validation error records.

    Returns:
        pandas.DataFrame: Processed DataFrame with mean and standard deviation for each K.
    """
    cv_loglikelihood_df = pd.DataFrame.from_records(cv_loglikelihood)
    cv_loglikelihood_reduced = pd.DataFrame(cv_loglikelihood_df.mean(), columns=["cv_loglikelihood_mean"])
    cv_loglikelihood_reduced["K"] = cv_loglikelihood_reduced.index.copy()
    cv_loglikelihood_reduced["cv_loglikelihood_std"] = cv_loglikelihood_df.std()
    cv_loglikelihood_reduced = cv_loglikelihood_reduced.sort_values("K")
    
    return cv_loglikelihood_reduced

def save_cv_error_plot(cv_loglikelihood_reduced: pd.DataFrame, save_dir: str) -> None:
    """
    Create and save a plot of cross-validation loglikelihood against K.

    Args:
        cv_loglikelihood_reduced (pandas.DataFrame): DataFrame with mean and standard deviation of cross-validation loglikelihoods.
        save_dir (str): Directory where the plot should be saved.

    Returns:
        None
    """
    # Conditional import for plotting if not already done or if in a minimal environment
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("    Seaborn or Matplotlib not found. Skipping CV error plot generation.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    # Ensure 'K' and 'cv_loglikelihood_mean' are numeric for plotting
    cv_loglikelihood_reduced['K'] = pd.to_numeric(cv_loglikelihood_reduced['K'])
    cv_loglikelihood_reduced['cv_loglikelihood_mean'] = pd.to_numeric(cv_loglikelihood_reduced['cv_loglikelihood_mean'])
    
    loglikelihood_plot = sns.lineplot(
        x='K', y='cv_loglikelihood_mean', data=cv_loglikelihood_reduced, marker='o'
        # err_style="bars" # This requires std dev data if you want error bars
    )
    # If you have std dev and want error bars:
    # plt.errorbar(cv_loglikelihood_reduced['K'], cv_loglikelihood_reduced['cv_loglikelihood_mean'], 
    #              yerr=cv_loglikelihood_reduced['cv_loglikelihood_std'], fmt='-o')


    loglikelihood_plot.set_title('Cross-validation Log Likelihood vs K', fontsize=18)
    loglikelihood_plot.set_xlabel('K (Number of Clusters)', fontsize=14)
    loglikelihood_plot.set_ylabel('Cross-validation Log Likelihood', fontsize=14)
    plt.xticks(cv_loglikelihood_reduced['K'].unique(), fontsize=12) # Ensure all K values are ticks
    plt.yticks(fontsize=12)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plot_file_name = Path(save_dir) / 'cv_loglikelihood_plot.png'
    plt.savefig(plot_file_name)
    plt.close()
    log.info(f"    Cross-validation plot saved to: {plot_file_name}")
    
def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For all GPUs
    np.random.seed(seed)
    random.seed(seed)
    # The following can impact performance, use if strict reproducibility is paramount
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
