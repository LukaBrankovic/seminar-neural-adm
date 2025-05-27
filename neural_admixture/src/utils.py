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
import time # Added time import

from pathlib import Path
from typing import List, Tuple, Union, Optional # Added Optional

from .snp_reader import SNPReader
from ..model.switchers import Switchers
from .ipca_gpu import GPUIncrementalPCA # Corrected import for type hinting

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_train_args(argv: List[str]):
    parser = configargparse.ArgumentParser(prog='neural-admixture train',
                                           description='Rapid population clustering with autoencoders - training mode',
                                           config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           allow_abbrev=False)
    
    parser.add_argument('--epochs', required=False, type=int, default=250, help='Maximum number of epochs.')
    parser.add_argument('--batch_size', required=False, default=800, type=int, help='Batch size.')
    parser.add_argument('--learning_rate', required=False, default=25e-4, type=float, help='Learning rate.')
    parser.add_argument('--initialization', required=False, type=str, default = 'gmm',
                        choices=['kmeans', 'gmm', 'supervised', 'random'], help='P initialization method.')
    parser.add_argument('--activation', required=False, default='relu', type=str, choices=['relu', 'tanh', 'gelu'], help='Activation function for encoder layers.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed for reproducibility.')
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters. Not required if --auto_k_dbscan is used, but can serve as a fallback if DBSCAN fails to find clusters.')
    parser.add_argument('--hidden_size', required=False, default=256, type=int, help='Dimension of the hidden layer in the encoder.')
    parser.add_argument('--pca_path', required=False, type=str, help='Path to store/load PCA object for DBSCAN or explicit initialization. If not provided, it will be constructed relative to save_dir (e.g., save_dir/name_pca.pt or save_dir/name_pca_for_dbscan.pt).')
    parser.add_argument('--pca_components', required=False, type=int, default=8, help='Number of principal components to use for PCA-based initializations and DBSCAN if auto_k_dbscan is enabled.')
    parser.add_argument('--save_dir', required=True, type=str, help='Directory to save model outputs, logs, and PCA object (if pca_path is not specified).')
    parser.add_argument('--data_path', required=True, type=str, help='Path to the input genotype data file (e.g., VCF, BED, PGEN, HDF5, NPY).')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name, used for naming output files.')
    parser.add_argument('--imputation', type=str, default='mean', choices=['mean', 'zero'], help='Imputation method for missing genotype data (default: mean).')
    parser.add_argument('--supervised_loss_weight', required=False, default=0.05, type=float, help='Weight for the supervised classification loss component if --supervised is used. Default: 0.05')
    parser.add_argument('--populations_path', required=False, default='', type=str, help='Path to a file containing population labels for supervised mode. Each line corresponds to a sample.')
    parser.add_argument('--supervised', action='store_true', default=False, help='Enable supervised mode. Requires --populations_path.')
    parser.add_argument('--num_gpus', required=False, default=0, type=int, help='Number of GPUs to use for training. Set to 0 for CPU-only.')
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPU cores to use (e.g., for DBSCAN n_jobs, DataLoader num_workers).')
    parser.add_argument('--auto_k_dbscan', action='store_true', default=False, help='Automatically determine K using DBSCAN on PCA components instead of using a fixed K.')
    parser.add_argument('--dbscan_eps', required=False, type=float, default=0.5, help='DBSCAN eps parameter. Used if --auto_k_dbscan is enabled.')
    parser.add_argument('--dbscan_min_samples', required=False, type=int, default=5, help='DBSCAN min_samples parameter. Used if --auto_k_dbscan is enabled.')
    
    args = parser.parse_args(argv)

    if args.auto_k_dbscan and args.k is not None:
        log.info("    Both --auto_k_dbscan and --k are specified. --k will be used as a fallback if DBSCAN fails or finds 0 clusters.")
    elif not args.auto_k_dbscan and args.k is None:
        parser.error("--k is required if --auto_k_dbscan is not used.")
    if args.supervised and not args.populations_path:
        parser.error("--supervised mode requires --populations_path to be specified.")
    if args.k is not None and args.k <=0:
        parser.error("--k must be a positive integer.")
    if args.pca_components <=0:
        parser.error("--pca_components must be a positive integer.")

    return args

def parse_infer_args(argv: List[str]):
    parser = configargparse.ArgumentParser(prog='neural-admixture infer',
                                     description='Rapid population clustering with autoencoders - inference mode',
                                     config_file_parser_class=configargparse.YAMLConfigFileParser,
                                     allow_abbrev=False)
    parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode.')
    parser.add_argument('--save_dir', required=True, type=str, help='Load model from this directory (where .pt, _config.json, _pca.pt files are).')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data for inference.')
    parser.add_argument('--name', required=True, type=str, help='Trained experiment/model name (used to find .pt, _config.json, _pca.pt).')
    parser.add_argument('--batch_size', required=False, default=1000, type=int, help='Batch size for inference.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed for reproducibility (primarily affects dataloader shuffling if any, less critical for inference).')
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPUs to be used for DataLoader workers during inference.')
    parser.add_argument('--imputation', type=str, default='mean', choices=['mean', 'zero'], help='Imputation method for missing genotype data (default: mean). Should match training if possible.')

    return parser.parse_args(argv)

def read_data(tr_file: str, master: bool, tr_pops_f: Optional[str]=None, imputation: str='mean') -> Tuple[da.core.Array, Union[List[str], None]]:
    snp_reader = SNPReader()
    data = snp_reader.read_data(tr_file, imputation, master)
    if master:
        log.info(f"    Data contains {data.shape[0]} samples and {data.shape[1]} SNPs.")
    
    tr_pops_list = None
    if tr_pops_f:
        if not os.path.exists(tr_pops_f): # Check if file exists before trying to open
            if master: log.error(f"    Populations file not found: {tr_pops_f}")
            sys.exit(1) # Exit if file not found
        try:
            with open(tr_pops_f, 'r') as fb:
                tr_pops_list = [p.strip() for p in fb.readlines()]
            if master:
                log.info(f"    Read {len(tr_pops_list)} population labels from {tr_pops_f}.")
            if len(tr_pops_list) != data.shape[0]:
                if master: log.error(f"    Mismatch between number of samples in data ({data.shape[0]}) and population labels ({len(tr_pops_list)}). Please ensure they match.")
                sys.exit(1) # Exit on mismatch
        except Exception as e: # Catch other potential errors during file reading
            if master: log.error(f"    Error reading populations file {tr_pops_f}: {e}")
            sys.exit(1)
    
    return data, tr_pops_list

def initialize_data(master: bool, trX_dask: da.core.Array, tr_pops_list: Union[List[str], None]=None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    if master:
        log.info("    Bringing data into memory (computing Dask array)...")
    
    t_start_compute = time.time()
    data_np = trX_dask.compute() 
    t_end_compute = time.time()

    if master:
        log.info(f"    Data successfully loaded into memory. Shape: {data_np.shape}. Time taken: {t_end_compute - t_start_compute:.2f}s.")
        log.info("")

    y_np = None
    if tr_pops_list is not None:
        y_np = np.array(tr_pops_list)
        
    return data_np, y_np


def train(initialization_method: str, device: torch.device, save_dir : str, name: str, 
        k_val: int, seed: int, pca_n_components: int, epochs: int, batch_size: int, learning_rate: float, 
        data_np: np.ndarray, num_gpus: int, activation_str: str, hidden_size: int, master: bool, num_cpus: int,
        y_np: Union[np.ndarray, None], supervised_loss_weight: float,
        precomputed_X_pca_tensor: Optional[torch.Tensor] = None,
        precomputed_pca_obj: Optional[GPUIncrementalPCA] = None
        ) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    """
    Train the model using specified initialization, hyperparameters, and data.
    Accepts pre-computed PCA results to ensure consistency.
    """
    init_pca_obj_filename = f'{name}_pca.pt' 
    init_pca_obj_path_for_init_method = str(Path(save_dir) / init_pca_obj_filename)
    
    switchers = Switchers.get_switchers()
    activation_fn = switchers['activations'][activation_str](0)

    P_np, Q_np, raw_model_instance = switchers['initializations'][initialization_method](
        epochs, batch_size, learning_rate, k_val, seed, 
        init_pca_obj_path_for_init_method, 
        name, pca_n_components, 
        data_np, 
        device, 
        num_gpus, hidden_size, activation_fn, master, num_cpus, 
        y_np, 
        supervised_loss_weight,
        precomputed_X_pca_tensor=precomputed_X_pca_tensor,
        precomputed_pca_obj=precomputed_pca_obj
    )
    
    return P_np, Q_np, raw_model_instance

def write_outputs(Q: np.ndarray, run_name: str, K_val: int, out_path_str: str, P: Optional[np.ndarray]=None) -> None:
    out_path = Path(out_path_str)
    out_path.mkdir(parents=True, exist_ok=True) 
    
    q_filename = out_path / f"{run_name}.{K_val}.Q"
    np.savetxt(q_filename, Q, delimiter=' ')
    log.info(f"    Q matrix saved to: {q_filename}")

    if P is not None:
        p_filename = out_path / f"{run_name}.{K_val}.P"
        np.savetxt(p_filename, P, delimiter=' ')
        log.info(f"    P matrix saved to: {p_filename}")
    return 

def find_free_port(start_port=12355, max_tries=1000): # Added max_tries
    port = start_port
    for _ in range(max_tries): # Limit number of tries
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port 
        except OSError: 
            port += 1
        if port > 65535: # Standard port range upper limit
            break # Avoid infinite loop if all ports are somehow busy
    raise RuntimeError(f"Could not find a free port after {max_tries} tries starting from {start_port}.")


def ddp_setup(stage: str, rank: int, world_size: int) -> None:
    if world_size <= 1: 
        if rank == 0: log.debug(f"    DDP Setup: Skipping DDP setup as world_size is {world_size}.")
        return

    if stage == 'begin':
        os.environ["MASTER_ADDR"] = "localhost"
        try:
            os.environ["MASTER_PORT"] = str(find_free_port()) 
        except RuntimeError as e: # Catch if find_free_port fails
            if rank == 0: log.error(f"    DDP Setup Error: {e}")
            raise # Re-raise to stop execution if port cannot be found
            
        backend_to_use = "nccl" 
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        elif torch.backends.mps.is_available(): 
            backend_to_use = "gloo" 
            if rank == 0: log.warning("    DDP Setup: Using MPS with 'gloo' backend. Multi-device DDP on MPS is not standard.")
        else: 
            backend_to_use = "gloo"
        
        if rank == 0: log.info(f"    DDP Setup: Initializing process group. Rank: {rank}, World Size: {world_size}, Backend: {backend_to_use}, Addr: {os.environ['MASTER_ADDR']}, Port: {os.environ['MASTER_PORT']}")
        
        try:
            torch.distributed.init_process_group(backend=backend_to_use, rank=rank, world_size=world_size)
            if rank == 0: log.info(f"    DDP Setup: Process group initialized for rank {rank}.")
        except RuntimeError as e: # Catch errors during init_process_group
            if rank == 0: log.error(f"    DDP Setup: Failed to initialize process group for rank {rank}: {e}")
            raise # Re-raise critical DDP setup error
    else: 
        if torch.distributed.is_initialized():
            if rank == 0: log.info(f"    DDP Setup: Destroying process group for rank {rank}.")
            torch.distributed.destroy_process_group()
            if rank == 0: log.info(f"    DDP Setup: Process group destroyed for rank {rank}.")
        else:
            if rank == 0: log.debug(f"    DDP Setup: Attempted to destroy uninitialized process group for rank {rank}. Skipping.")

def process_cv_loglikelihood(cv_loglikelihood: list) -> pd.DataFrame:
    if not cv_loglikelihood: 
        log.warning("    CV loglikelihood list is empty. Cannot process.")
        return pd.DataFrame(columns=["K", "cv_loglikelihood_mean", "cv_loglikelihood_std"])

    cv_loglikelihood_df = pd.DataFrame.from_records(cv_loglikelihood)
    
    if 'K' not in cv_loglikelihood_df.columns or not any(col.startswith('cv_loglikelihood') or col == 'cv_score' for col in cv_loglikelihood_df.columns):
        log.warning("    CV loglikelihood DataFrame does not have expected 'K' and score columns. Cannot process correctly.")
        return pd.DataFrame({'K': [], 'cv_loglikelihood_mean': [], 'cv_loglikelihood_std': []})

    score_col = 'cv_loglikelihood' 
    if score_col not in cv_loglikelihood_df.columns:
        possible_score_cols = [col for col in cv_loglikelihood_df.columns if 'loglikelihood' in col or 'score' in col and col != 'K']
        if not possible_score_cols:
            log.error("    Cannot find a score column in CV data.")
            return pd.DataFrame({'K': [], 'cv_loglikelihood_mean': [], 'cv_loglikelihood_std': []})
        score_col = possible_score_cols[0]
        if master: log.info(f"    Using '{score_col}' as the score column for CV processing.") # master not defined here, remove if master check needed

    cv_loglikelihood_reduced = cv_loglikelihood_df.groupby('K')[score_col].agg(['mean', 'std']).reset_index()
    cv_loglikelihood_reduced.rename(columns={'mean': 'cv_loglikelihood_mean', 'std': 'cv_loglikelihood_std'}, inplace=True)
    cv_loglikelihood_reduced = cv_loglikelihood_reduced.sort_values("K")
    
    return cv_loglikelihood_reduced

def save_cv_error_plot(cv_loglikelihood_reduced: pd.DataFrame, save_dir_str: str) -> None:
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("    Seaborn or Matplotlib not found. Skipping CV error plot generation.")
        return

    if cv_loglikelihood_reduced.empty or 'K' not in cv_loglikelihood_reduced.columns or 'cv_loglikelihood_mean' not in cv_loglikelihood_reduced.columns:
        log.warning("    CV data for plotting is empty or missing required columns (K, cv_loglikelihood_mean). Skipping plot.")
        return

    save_dir = Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    cv_loglikelihood_reduced['K'] = pd.to_numeric(cv_loglikelihood_reduced['K'])
    cv_loglikelihood_reduced['cv_loglikelihood_mean'] = pd.to_numeric(cv_loglikelihood_reduced['cv_loglikelihood_mean'])
    
    lineplot = sns.lineplot(
        x='K', y='cv_loglikelihood_mean', data=cv_loglikelihood_reduced, marker='o',
        hue=None, size=None, style=None, legend=False 
    )
    
    if 'cv_loglikelihood_std' in cv_loglikelihood_reduced.columns and cv_loglikelihood_reduced['cv_loglikelihood_std'].notna().any():
        cv_loglikelihood_reduced['cv_loglikelihood_std'] = pd.to_numeric(cv_loglikelihood_reduced['cv_loglikelihood_std'], errors='coerce').fillna(0)
        plt.errorbar(
            cv_loglikelihood_reduced['K'], 
            cv_loglikelihood_reduced['cv_loglikelihood_mean'], 
            yerr=cv_loglikelihood_reduced['cv_loglikelihood_std'], 
            fmt='none', 
            ecolor='gray', elinewidth=1, capsize=3, alpha=0.7
        )

    lineplot.set_title('Cross-validation Log Likelihood vs K', fontsize=16)
    lineplot.set_xlabel('K (Number of Ancestral Populations)', fontsize=14)
    lineplot.set_ylabel('Cross-validation Log Likelihood', fontsize=14)
    
    k_values = sorted(cv_loglikelihood_reduced['K'].unique())
    if k_values:
        plt.xticks(ticks=k_values, labels=[str(int(k)) for k in k_values], fontsize=12)
    
    plt.yticks(fontsize=12)
    plt.tight_layout() 
    
    plot_file_name = save_dir / 'cv_loglikelihood_plot.png'
    try:
        plt.savefig(plot_file_name)
        log.info(f"    Cross-validation plot saved to: {plot_file_name}")
    except Exception as e:
        log.error(f"    Failed to save CV plot: {e}")
    finally:
        plt.close()

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    log.info(f"    Global seed set to {seed}")