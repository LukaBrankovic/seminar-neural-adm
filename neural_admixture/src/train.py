import argparse
import dask.array as da
import logging
import sys
import time
import torch
from pathlib import Path
# from sklearn.model_selection import KFold # Original CV import, commented out as CV logic is not implemented here
from typing import List, Union # Added Union
from argparse import ArgumentError, ArgumentTypeError
# from pathlib import Path # Already imported

from . import utils # utils.py in the same directory

# Imports for DBSCAN and PCA
from sklearn.cluster import DBSCAN
from neural_admixture.model.initializations import load_or_compute_pca # Import the PCA function

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, trX_dask: da.core.Array, device: torch.device, 
              num_gpus_for_ddp: int, tr_pops_list: Union[List[str], None], 
              master: bool, rank: int) -> None:
    """
    Wrapper function to start training.
    Determines K using DBSCAN if specified, then trains the Neural ADMIXTURE model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        trX_dask (da.core.Array): Training data as a Dask array.
        device (torch.device): PyTorch device for the current process (e.g., 'cuda:0', 'cpu').
        num_gpus_for_ddp (int): Number of GPUs used for DDP (world size).
        tr_pops_list (Union[List[str], None]): List of population labels or None.
        master (bool): True if this is the master process (rank 0).
        rank (int): Rank of the current DDP process.
    """
    # Extract necessary parameters from args
    epochs = int(args.epochs)
    batch_size_arg = int(args.batch_size) # Will be divided by num_gpus if DDP
    learning_rate = float(args.learning_rate)
    save_dir = args.save_dir
    activation_str = args.activation
    hidden_size = int(args.hidden_size)
    initial_initialization_method = args.initialization
    pca_n_components_arg = int(args.pca_components)
    experiment_name = args.name
    seed = int(args.seed)
    supervised_loss_weight = float(args.supervised_loss_weight)
    num_cpus_arg = int(args.num_cpus)
        
    utils.set_seed(seed) # Set seed for reproducibility
    
    # Convert Dask array to NumPy array (potentially memory intensive)
    # This also converts population list to NumPy array if present.
    data_np, y_np = utils.initialize_data(master, trX_dask, tr_pops_list)

    K_to_use = None # This will store the final K value for training

    if not args.auto_k_dbscan:
        if args.k is None: # This case should be caught by argparser, but defensive check
            if master:
                log.error("Critical Error: --k must be specified if --auto_k_dbscan is not used. Argparser should have caught this.")
            # utils.ddp_setup('end', rank, num_gpus_for_ddp) # Cleanup before exit
            sys.exit(1) # Exit if K is somehow None here
        K_to_use = int(args.k)
        if master:
            log.info(f"    Using user-specified K = {K_to_use}.")
    else: # --auto_k_dbscan is True
        if master:
            log.info(f"    Attempting to determine K using DBSCAN on PCA components (eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}).")

        # Determine PCA object path for DBSCAN
        if args.pca_path: # User specified a path for PCA object
            pca_obj_path_for_dbscan_str = args.pca_path
        else: # Construct path based on save_dir and name
            pca_obj_filename_for_dbscan = f'{experiment_name}_pca_for_dbscan.pt' # Use a distinct name if needed
            pca_obj_path_for_dbscan_str = str(Path(save_dir) / pca_obj_filename_for_dbscan)
        
        Path(pca_obj_path_for_dbscan_str).parent.mkdir(parents=True, exist_ok=True)

        # Perform PCA. `load_or_compute_pca` is from `initializations.py`.
        # It handles loading if exists, or computing and saving.
        # For DBSCAN, PCA is typically done on CPU.
        pca_device_for_dbscan = torch.device('cpu') 
        if master:
            log.info(f"    Computing/Loading PCA for DBSCAN on device: {pca_device_for_dbscan} with {pca_n_components_arg} components.")
            log.info(f"    PCA object for DBSCAN will be saved/loaded from: {pca_obj_path_for_dbscan_str}")

        # `data_np` is the NumPy array of genotype data
        X_pca_tensor, _ = load_or_compute_pca(
            path=pca_obj_path_for_dbscan_str,
            X=data_np,
            n_components=pca_n_components_arg,
            batch_size=1024, # A reasonable batch size for IncrementalPCA
            device=pca_device_for_dbscan,
            run_name=f"{experiment_name}_dbscan_pca_plot", # For PCA plot if generated
            master=master,
            sample_fraction=1.0 # Use full data for this PCA
        )
        X_pca_for_dbscan_np = X_pca_tensor.numpy() # Convert to NumPy for scikit-learn

        if master:
            log.info(f"    PCA for DBSCAN computed/loaded. Shape of PCA output: {X_pca_for_dbscan_np.shape}")
            log.info(f"    Running DBSCAN...")

        dbscan_clustering = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=num_cpus_arg)
        labels = dbscan_clustering.fit_predict(X_pca_for_dbscan_np)
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0) # Exclude noise points

        if n_clusters_found > 0:
            if master:
                log.info(f"    DBSCAN found {n_clusters_found} clusters. Using this as K.")
            K_to_use = n_clusters_found
        else: # Fallback if DBSCAN fails
            if master:
                log.warning(f"    DBSCAN found 0 clusters or only noise. This might indicate unsuitable DBSCAN parameters (eps, min_samples) for your data, or issues with data quality/structure after PCA.")
            if args.k is not None: # Fallback to user-provided --k if available
                K_to_use = int(args.k)
                if master:
                    log.warning(f"    Falling back to user-specified K = {K_to_use}.")
            else:
                if master:
                    log.error("    DBSCAN failed to find clusters, and no fallback --k was provided. Exiting.")
                # utils.ddp_setup('end', rank, num_gpus_for_ddp) # Cleanup
                sys.exit(1)
    
    # Determine the final initialization method based on args.supervised
    current_initialization_method = initial_initialization_method
    if bool(args.supervised):
        current_initialization_method = 'supervised'
        if master:
            log.info(f"    Running in supervised mode. Initialization method set to 'supervised'.")
            if args.auto_k_dbscan:
                 log.warning(f"    --auto_k_dbscan was used with --supervised. K determined by DBSCAN ({K_to_use}) will be used. "
                             f"Ensure this K value is consistent with the number of unique populations in your labels file ({args.populations_path}) if that's your intent.")
        if y_np is None: # Check if population data (y_np) is actually available from --populations_path
             if master:
                log.error(f"    Supervised mode selected, but no population labels were loaded (from --populations_path). "
                          f"Please provide a valid populations file. Exiting.")
             # utils.ddp_setup('end', rank, num_gpus_for_ddp) # Cleanup
             sys.exit(1)
        # Further validation: Compare K_to_use with number of unique labels in y_np
        if y_np is not None:
            unique_supervised_labels = len(set(label for label in y_np if label != '-')) # Exclude missing label indicator
            if K_to_use != unique_supervised_labels:
                if master:
                    log.warning(f"    Mismatch in K for supervised mode: K set to {K_to_use} (by DBSCAN or --k), "
                                f"but found {unique_supervised_labels} unique populations in the labels file. "
                                f"The model will be trained for K={K_to_use}. Ensure this is intended.")


    if master:
        log.info(f"    Proceeding to train Neural ADMIXTURE model with K = {K_to_use}.")
        
    # Call utils.train with the determined K_to_use and other parameters
    # Note: `data_np` (NumPy array) and `y_np` (NumPy array or None) are passed to utils.train
    # `pca_n_components_arg` is passed for the initialization's internal PCA needs.
    P_matrix_np, Q_matrix_np, trained_model_instance = utils.train(
        initialization_method=current_initialization_method, 
        device=device, 
        save_dir=save_dir, 
        name=experiment_name, 
        k_val=K_to_use, 
        seed=seed, 
        pca_n_components=pca_n_components_arg, # PCA components for the initialization strategy
        epochs=epochs, 
        batch_size=batch_size_arg, # utils.train's internal model will adjust for DDP
        learning_rate=learning_rate, 
        data_np=data_np, 
        num_gpus=num_gpus_for_ddp, # For DDP setup within the model
        activation_str=activation_str, 
        hidden_size=hidden_size, 
        master=master, 
        num_cpus=num_cpus_arg, # For DataLoader workers
        y_np=y_np, 
        supervised_loss_weight=supervised_loss_weight
    )
    
    if master: # Only master process saves files
        Path(save_dir).mkdir(parents=True, exist_ok=True) # Ensure save directory exists
        
        # Save the model state (excluding the P matrix, which is saved separately)
        model_save_path = Path(save_dir) / f'{experiment_name}.pt'
        # `trained_model_instance` is the raw Q_P model instance
        state_dict_to_save = {k: v for k, v in trained_model_instance.state_dict().items() if k != 'P'}
        torch.save(state_dict_to_save, model_save_path)
        log.info(f"    Model state (encoder weights) saved to: {model_save_path}")
        
        # Save model configuration using the K it was trained with
        trained_model_instance.save_config(experiment_name, save_dir) 
        
        # Save Q and P matrices
        utils.write_outputs(Q_matrix_np, experiment_name, K_to_use, save_dir, P_matrix_np)

    return

def main(rank: int, argv: List[str], num_gpus_from_entry: int):
    """
    Main training entry point, called by `neural_admixture.entry.py`.
    Handles DDP setup, argument parsing, and calls `fit_model`.

    Args:
        rank (int): Rank of the current DDP process.
        argv (List[str]): Command-line arguments.
        num_gpus_from_entry (int): Number of GPUs for DDP (world size), passed from entry.py.
    """
    # Initialize DDP. num_gpus_from_entry is the world_size for DDP.
    utils.ddp_setup('begin', rank, num_gpus_from_entry)
    master = (rank == 0) # True if this is the master process
    
    try:
        # Handle --help explicitly to avoid parsing errors in non-master processes
        if any(arg in argv for arg in ['-h', '--help']):
            if master:
                # Allow argparse to handle help and exit
                _ = utils.parse_train_args(argv) 
            # For non-master, just wait for master to exit or DDP to clean up
            # utils.ddp_setup('end', rank, num_gpus_from_entry) # Cleanup handled in finally
            return 
            
        args = utils.parse_train_args(argv) # Parse arguments
        
        # Determine the torch device for this DDP process
        if num_gpus_from_entry > 0 and torch.cuda.is_available():
            # DDP requires each process to be on a specific GPU
            device = torch.device(f'cuda:{rank}')
        elif num_gpus_from_entry > 0 and torch.backends.mps.is_available(): # MPS for single "GPU"
            device = torch.device('mps')
            if num_gpus_from_entry > 1 and master:
                 log.warning("    MPS backend does not support multi-GPU DDP effectively. Running as if on a single device.")
        else: # CPU
            device = torch.device('cpu')
            if num_gpus_from_entry > 0 and master: # User asked for GPUs but none available
                log.warning("    Requested GPUs but none are available (CUDA/MPS). Falling back to CPU.")


        if master:
            log.info(f"    Process Rank: {rank} (Master Process)")
            log.info(f"    Number of GPUs for DDP (World Size): {num_gpus_from_entry}")
            log.info(f"    Torch device for this process: {device}")
            log.info(f"    Number of CPUs specified by --num_cpus: {args.num_cpus} (used for DBSCAN n_jobs, DataLoader workers, etc.)")
            log.info("")
            Path(args.save_dir).mkdir(parents=True, exist_ok=True) # Ensure save directory exists
        
        start_time = time.time()
        
        # Read data (returns Dask array and possibly population labels list)
        trX_dask, tr_pops_list = utils.read_data(
            args.data_path, 
            master, 
            args.populations_path if args.supervised else None, # Only pass pop path if supervised
            args.imputation
        )

        # Call the main model fitting function
        fit_model(args, trX_dask, device, num_gpus_from_entry, tr_pops_list, master, rank)
        
        if master:
            end_time = time.time()
            log.info("")
            log.info(f"    Total training execution time: {end_time - start_time:.2f} seconds.")
            log.info("")
        
    except SystemExit: # Allows sys.exit() to terminate script cleanly
        if master:
            log.info("    Exiting training process due to sys.exit() call.")
    except (ArgumentError, ArgumentTypeError) as e: # Handle argparse errors
        if master:
            log.error(f"    Argument Parsing Error: {e}")
            # Argparse usually prints help and exits, but if not, we exit here.
        # sys.exit(1) # Let finally handle DDP cleanup
    except Exception as e: # Catch any other unexpected errors
        if master:
            log.error(f"    An unexpected error occurred during training:", exc_info=True) # Log traceback
        # sys.exit(1) # Let finally handle DDP cleanup
    finally:
        # Ensure DDP process group is always destroyed
        logging.shutdown() # Flush and close all handlers
        utils.ddp_setup('end', rank, num_gpus_from_entry)
        
if __name__ == '__main__':
    # This script (`train.py`) is typically not run directly with DDP.
    # The `neural_admixture.entry:main` function handles spawning DDP processes,
    # which then call this `main(rank, argv, num_gpus_from_entry)` function.
    # For local testing without DDP, you might call it with rank=0, num_gpus_from_entry=0 or 1.
    # e.g., main(0, sys.argv[1:], 0)
    pass