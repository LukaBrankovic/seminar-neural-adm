import argparse
import dask.array as da
import logging
import sys
import time
import torch
from pathlib import Path
from typing import List, Union, Optional 

from . import utils 
from neural_admixture.model.initializations import load_or_compute_pca 
from .ipca_gpu import GPUIncrementalPCA 

from sklearn.cluster import DBSCAN


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, trX_dask: da.core.Array, device: torch.device, 
              num_gpus_for_ddp: int, tr_pops_list: Union[List[str], None], 
              master: bool, rank: int) -> None:
    """
    Wrapper function to start training.
    Determines K using DBSCAN if specified, then trains the Neural ADMIXTURE model,
    reusing PCA from DBSCAN step for initialization if auto_k_dbscan is active.
    """
    epochs = int(args.epochs)
    batch_size_arg = int(args.batch_size) 
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
        
    utils.set_seed(seed) 
    
    data_np, y_np = utils.initialize_data(master, trX_dask, tr_pops_list)

    K_to_use: Optional[int] = None 
    X_pca_tensor_for_init: Optional[torch.Tensor] = None
    pca_obj_for_init: Optional[GPUIncrementalPCA] = None

    if not args.auto_k_dbscan:
        if args.k is None: 
            if master: log.error("Critical Error: --k must be specified if --auto_k_dbscan is not used.")
            sys.exit(1) 
        K_to_use = int(args.k)
        if master: log.info(f"    Using user-specified K = {K_to_use}.")
    else: 
        if master:
            log.info(f"    Attempting to determine K using DBSCAN on PCA (eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}, pca_components={pca_n_components_arg}).")

        if args.pca_path: 
            pca_obj_path_for_dbscan_and_init_str = args.pca_path
            if master: log.info(f"    User-specified --pca_path '{args.pca_path}' will be used for DBSCAN's PCA and reused for initialization.")
        else: 
            pca_obj_filename_for_dbscan_and_init = f'{experiment_name}_pca_for_dbscan_reused.pt' 
            pca_obj_path_for_dbscan_and_init_str = str(Path(save_dir) / pca_obj_filename_for_dbscan_and_init)
            if master: log.info(f"    PCA for DBSCAN (and reused for initialization) will be at: {pca_obj_path_for_dbscan_and_init_str}")
        
        Path(pca_obj_path_for_dbscan_and_init_str).parent.mkdir(parents=True, exist_ok=True)

        pca_device_for_dbscan = torch.device('cpu') 
        if master:
            log.info(f"    Computing/Loading PCA for DBSCAN on device: {pca_device_for_dbscan} with {pca_n_components_arg} components.")

        X_pca_tensor_for_init, pca_obj_for_init = load_or_compute_pca(
            path=pca_obj_path_for_dbscan_and_init_str, 
            X=data_np,
            n_components=pca_n_components_arg,
            batch_size=1024, 
            device=pca_device_for_dbscan, 
            run_name=f"{experiment_name}_dbscan_reused_pca", 
            master=master,
            sample_fraction=1.0 
        )
        
        # Convert to float32 before converting to NumPy
        X_pca_for_dbscan_np = X_pca_tensor_for_init.cpu().to(torch.float32).numpy()

        if master:
            log.info(f"    PCA for DBSCAN computed/loaded. Shape: {X_pca_for_dbscan_np.shape}. PCA object is on {pca_obj_for_init.device}.")
            log.info(f"    Running DBSCAN...")

        dbscan_clustering = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=num_cpus_arg)
        labels = dbscan_clustering.fit_predict(X_pca_for_dbscan_np)
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0) 

        if n_clusters_found > 0:
            if master: log.info(f"    DBSCAN found {n_clusters_found} clusters. Using this as K.")
            K_to_use = n_clusters_found
        else: 
            if master: log.warning(f"    DBSCAN found 0 clusters or only noise.")
            if args.k is not None: 
                K_to_use = int(args.k)
                if master: log.warning(f"    Falling back to user-specified K = {K_to_use}.")
            else:
                if master: log.error("    DBSCAN failed, and no fallback --k provided. Exiting.")
                sys.exit(1)
    
    current_initialization_method = initial_initialization_method
    if bool(args.supervised):
        current_initialization_method = 'supervised'
        if master:
            log.info(f"    Running in supervised mode. Initialization method forced to 'supervised'.")
            if args.auto_k_dbscan:
                 log.warning(f"    --auto_k_dbscan was used with --supervised. K determined by DBSCAN ({K_to_use}) will be used.")
        if y_np is None: 
             if master: log.error(f"    Supervised mode selected, but no population labels loaded. Exiting.")
             sys.exit(1)
        if y_np is not None: 
            unique_supervised_labels = len(set(label for label in y_np if label != '-')) 
            if K_to_use != unique_supervised_labels:
                if master:
                    log.warning(f"    Mismatch in K for supervised mode: K is {K_to_use} (from DBSCAN/--k), "
                                f"but found {unique_supervised_labels} unique populations in labels file. "
                                f"Model trains for K={K_to_use}. P_init in supervised init will be based on {unique_supervised_labels} labels and then adjusted if K differs.")

    if K_to_use is None or K_to_use <= 0: 
        if master: log.error(f"    Invalid K value determined: {K_to_use}. K must be a positive integer. Exiting.")
        sys.exit(1)

    if master:
        log.info(f"    Proceeding to train Neural ADMIXTURE model with K = {K_to_use} using '{current_initialization_method}' initialization.")
        
    train_call_params = {
        'initialization_method': current_initialization_method, 
        'device': device,  
        'save_dir': save_dir, 
        'name': experiment_name, 
        'k_val': K_to_use, 
        'seed': seed, 
        'pca_n_components': pca_n_components_arg, 
        'epochs': epochs, 
        'batch_size': batch_size_arg,
        'learning_rate': learning_rate, 
        'data_np': data_np, 
        'num_gpus': num_gpus_for_ddp,
        'activation_str': activation_str, 
        'hidden_size': hidden_size, 
        'master': master, 
        'num_cpus': num_cpus_arg,
        'y_np': y_np, 
        'supervised_loss_weight': supervised_loss_weight,
        'precomputed_X_pca_tensor': X_pca_tensor_for_init, 
        'precomputed_pca_obj': pca_obj_for_init           
    }
    
    P_matrix_np, Q_matrix_np, trained_model_instance = utils.train(**train_call_params)
    
    if master: 
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model_save_path = Path(save_dir) / f'{experiment_name}.pt'
        torch.save(trained_model_instance.state_dict(), model_save_path)
        log.info(f"    Model state_dict saved to: {model_save_path}")
        
        trained_model_instance.save_config(experiment_name, save_dir) 
        utils.write_outputs(Q_matrix_np, experiment_name, K_to_use, save_dir, P_matrix_np)

    return

def main(rank: int, argv: List[str], num_gpus_from_entry: int):
    utils.ddp_setup('begin', rank, num_gpus_from_entry)
    master = (rank == 0) 
    
    current_device: torch.device
    if num_gpus_from_entry > 0 and torch.cuda.is_available():
        current_device = torch.device(f'cuda:{rank}')
    elif num_gpus_from_entry > 0 and torch.backends.mps.is_available():
        current_device = torch.device('mps')
        if num_gpus_from_entry > 1 and master:
             log.warning("    MPS backend does not support multi-GPU DDP effectively. Running as if on a single device for DDP.")
    else: 
        current_device = torch.device('cpu')
        if num_gpus_from_entry > 0 and master: 
            log.warning("    Requested GPUs but none are available (CUDA/MPS). Falling back to CPU.")

    try:
        if any(arg in argv for arg in ['-h', '--help']):
            if master:
                _ = utils.parse_train_args(argv) 
            return 
            
        args = utils.parse_train_args(argv)
        
        if master:
            log.info(f"    Process Rank: {rank} (Master Process)")
            log.info(f"    Number of GPUs for DDP (World Size): {num_gpus_from_entry}")
            log.info(f"    Torch device for this process: {current_device}")
            log.info(f"    Number of CPUs for system tasks (DBSCAN, DataLoader): {args.num_cpus}")
            log.info("")
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        trX_dask, tr_pops_list = utils.read_data(
            args.data_path, 
            master, 
            args.populations_path if args.supervised else None,
            args.imputation
        )

        fit_model(args, trX_dask, current_device, num_gpus_from_entry, tr_pops_list, master, rank)
        
        if master:
            end_time = time.time()
            log.info("")
            log.info(f"    Total training execution time: {end_time - start_time:.2f} seconds.")
            log.info("")
        
    except SystemExit: 
        if master: log.info("    Exiting training process due to sys.exit() call.")
    except (argparse.ArgumentError, argparse.ArgumentTypeError) as e: 
        if master: log.error(f"    Argument Parsing Error: {e}")
    except Exception as e: 
        if master: log.error(f"    An unexpected error occurred during training:", exc_info=True)
    finally:
        logging.shutdown() 
        utils.ddp_setup('end', rank, num_gpus_from_entry)
        
if __name__ == '__main__':
    is_ddp_spawned = "LOCAL_RANK" in os.environ or "RANK" in os.environ
    if not is_ddp_spawned:
        temp_argv = sys.argv[1:]
        num_gpus_for_direct_run = 0
        if '--num_gpus' in temp_argv:
            try:
                idx = temp_argv.index('--num_gpus')
                num_gpus_for_direct_run = int(temp_argv[idx+1])
                if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                    num_gpus_for_direct_run = 0 
                elif num_gpus_for_direct_run > 1 and torch.backends.mps.is_available() and not torch.cuda.is_available():
                    num_gpus_for_direct_run = 1 
                elif num_gpus_for_direct_run > torch.cuda.device_count():
                     num_gpus_for_direct_run = torch.cuda.device_count()
            except (ValueError, IndexError):
                pass 
        
        if num_gpus_for_direct_run > 1 and torch.cuda.is_available():
             print("Direct execution of train.py with multi-GPU DDP is complex to simulate here.")
             print("Please use `neural-admixture train ...` (via entry.py) for multi-GPU DDP.")
             print("For single GPU/CPU direct test, ensure --num_gpus is 0 or 1.")
             num_gpus_for_direct_run = 1 if torch.cuda.is_available() and num_gpus_for_direct_run > 0 else 0

        print(f"Running train.py directly (not spawned by entry.py's DDP). Rank: 0, World Size: {max(1, num_gpus_for_direct_run)}")
        main(rank=0, argv=sys.argv[1:], num_gpus_from_entry=max(1, num_gpus_for_direct_run))
    else:
        pass