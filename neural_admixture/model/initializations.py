import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import os

from pathlib import Path
from typing import List, Tuple, Optional, Union 
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture as ScikitGaussianMixture 
from scipy.optimize import linear_sum_assignment as linear_assignment

from ..src.ipca_gpu import GPUIncrementalPCA 
from .neural_admixture import NeuralAdmixture

torch.serialization.add_safe_globals([GPUIncrementalPCA])

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def determine_device_for_tensors(data_shape: tuple, K: int, device: torch.device, memory_threshold: float = 0.9) -> torch.device:
    """
    Determine if tensors can fit in GPU memory and return appropriate device.
    """
    def bytes_to_human_readable(bytes_value: int) -> str:
        gb = bytes_value / (1024**3)
        if gb >= 1:
            return f"{gb:.2f} GB"
        mb = bytes_value / (1024**2)
        return f"{mb:.2f} MB"

    def calculate_tensor_memory(shape, dtype_itemsize=4) -> int: 
        num_elements = np.prod(shape)
        return num_elements * dtype_itemsize
    
    if 'cuda' in device.type and torch.cuda.is_available():
        try:
            device_properties = torch.cuda.get_device_properties(device)
            available_gpu_memory = device_properties.total_memory - torch.cuda.memory_reserved(device)
        except RuntimeError: 
            if torch.cuda.device_count() > 0:
                 device_properties = torch.cuda.get_device_properties(0) 
                 available_gpu_memory = device_properties.total_memory - torch.cuda.memory_reserved(0)
            else: 
                log.warning("    determine_device_for_tensors: CUDA reported available, but no devices found or error querying properties. Defaulting to CPU for tensors.")
                return torch.device('cpu')


        data_dtype_itemsize = 2 if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else 4

        pca_output_features = min(K, data_shape[1]) if K > 0 else data_shape[1]


        memory_required = {
            'P': calculate_tensor_memory((data_shape[1], K)), 
            'Q': calculate_tensor_memory((data_shape[0], K)), 
            'data_tensor_model': calculate_tensor_memory(data_shape, dtype_itemsize=data_dtype_itemsize),
            'input_tensor_model': calculate_tensor_memory((data_shape[0], pca_output_features)) 
        }
        
        total_memory_required = sum(memory_required.values())
        fits_in_gpu = total_memory_required <= (available_gpu_memory * memory_threshold)
        device_tensors = device if fits_in_gpu else torch.device('cpu')
        
        current_cuda_device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1
        target_device_index = device.index if device.type == 'cuda' and device.index is not None else -1

        if device.type == 'cpu' or (device.type == 'cuda' and current_cuda_device_index == target_device_index):
            # Only log if master process or relevant context
            # Assuming this function might be called by non-master, so defer detailed logging to caller if needed
            pass # log.info(f"    Available GPU memory on {device if device.type == 'cuda' else 'system (for CPU decision)'}: {bytes_to_human_readable(available_gpu_memory if device.type == 'cuda' else 0)}")
                 # log.info(f"    Estimated memory for model tensors: {bytes_to_human_readable(total_memory_required)}")
                 # log.info(f"    Tensors for NeuralAdmixture model training will be on device: {device_tensors}")

    elif 'mps' in device.type and torch.backends.mps.is_available():
        # log.info(f"    Using MPS device. Assuming tensors fit in unified memory.") # Defer logging
        device_tensors = device 
    else: 
        # log.info(f"    Using CPU device. Tensors will be on CPU.") # Defer logging
        device_tensors = torch.device('cpu')
        
    return device_tensors

def pca_plot(X_pca_np: np.ndarray, path: str) -> None: 
    if not isinstance(X_pca_np, np.ndarray):
        log.warning("    PCA data for plotting is not a NumPy array. Skipping plot.")
        return

    if X_pca_np.shape[1] < 2:
        log.warning("    PCA data has fewer than 2 components, cannot create 2D scatter plot.")
        return

    plt.figure(figsize=(15,10))
    plt.scatter(X_pca_np[:,0], X_pca_np[:,1], s=.9, c='black', alpha=0.6)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Data projected onto first two principal components', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    try:
        plt.savefig(path)
        log.info(f"    PCA plot saved to {path}")
    except Exception as e:
        log.error(f"    Failed to save PCA plot to {path}: {e}")
    finally:
        plt.close()
    return

def load_or_compute_pca(path: Optional[str], X: np.ndarray, n_components: int, batch_size: int, 
                        device: torch.device, run_name: str, master: bool, sample_fraction: Optional[float]=None
                        ) -> Tuple[torch.Tensor, GPUIncrementalPCA]:
    X_original_shape = X.shape 
    pca_obj: Optional[GPUIncrementalPCA] = None
    X_pca_tensor: Optional[torch.Tensor] = None
    
    try:
        if path and os.path.exists(path):
            if master: log.info(f"            Attempting to load PCA object from: {path}")
            pca_obj_loaded = torch.load(path, map_location=torch.device('cpu')) 
            
            if not isinstance(pca_obj_loaded, GPUIncrementalPCA):
                raise TypeError("Loaded object is not a GPUIncrementalPCA instance.")
            
            pca_obj = pca_obj_loaded.to(device) 
            if master: log.info(f"            PCA object loaded successfully from {path}. Moved to device: {pca_obj.device}")

            if pca_obj.n_features_in_ != X_original_shape[1]:
                if master: log.warning(f"            Loaded PCA n_features_in_ ({pca_obj.n_features_in_}) "
                                     f"mismatches data columns ({X_original_shape[1]}). Recomputing PCA.")
                raise FileNotFoundError 
            
            if master: log.info(f"            Transforming data using loaded PCA object (on device {pca_obj.device})...")
            X_pca_tensor = pca_obj.transform(torch.as_tensor(X, dtype=torch.float32)) 
            if master: log.info(f"            Data transformed. PCA output shape: {X_pca_tensor.shape}")
        else:
            if master and path: log.info(f"            PCA object not found at {path} or path not specified.")
            raise FileNotFoundError 
        
    except (FileNotFoundError, TypeError, AttributeError, AssertionError, RuntimeError) as e:
        if master: 
            if isinstance(e, FileNotFoundError) and path and os.path.exists(path): 
                 log.warning(f"            Error loading PCA object from {path} (Error: {e}). Recomputing PCA.")
            elif isinstance(e, FileNotFoundError):
                 log.info(f"            PCA file not found. Will compute new PCA.")
            else: 
                 log.warning(f"            Could not load or validate PCA object (Error: {e}). Recomputing PCA.")

        X_for_pca_fit = X
        if sample_fraction is not None and 0 < sample_fraction < 1:
            num_rows = X.shape[0]
            sampled_indices = np.random.choice(num_rows, size=int(sample_fraction * num_rows), replace=False)
            X_for_pca_fit = X[sampled_indices, :]
            if master: log.info(f"            Using {X_for_pca_fit.shape[0]}/{num_rows} samples to compute PCA.")
        
        if master: log.info(f"            Performing IncrementalPCA on device: {device} with {n_components} components...")
        
        pca_obj = GPUIncrementalPCA(n_components=int(n_components), batch_size=batch_size, device=device)
        X_tensor_for_fit = torch.as_tensor(X_for_pca_fit, dtype=torch.float32) 
        
        if X_for_pca_fit.shape[0] == X.shape[0]: 
            X_pca_tensor = pca_obj.fit_transform(X_tensor_for_fit)
        else: 
            pca_obj.fit(X_tensor_for_fit)
            X_pca_tensor = pca_obj.transform(torch.as_tensor(X, dtype=torch.float32))

        if master: log.info(f"            PCA computed. Output shape: {X_pca_tensor.shape}. PCA object device: {pca_obj.device}")

        if path and master: 
            try:
                torch.save(pca_obj, path) 
                log.info(f"            PCA object saved to: {path}")
            except Exception as ex_save:
                log.error(f"            Failed to save PCA object to {path}: {ex_save}")
    
    if master:
        try:
            plot_save_path_str: str
            if path: 
                 plot_save_path_str = str(Path(path).parent / f"{Path(path).stem.replace('_pca_for_dbscan_reused', '').replace('_pca_for_dbscan', '').replace('_pca', '')}_pca_plot.png")
            else: 
                 plot_save_path_str = str(Path(run_name).parent / f"{Path(run_name).name}_pca_plot.png")
            
            Path(plot_save_path_str).parent.mkdir(parents=True, exist_ok=True)
            pca_plot(X_pca_tensor.cpu().to(torch.float32).numpy(), plot_save_path_str)
        except Exception as e_plot:
            log.warning(f"            Could not render PCA plot: {e_plot}") 
    
    if X_pca_tensor.dtype == torch.bfloat16: # Should ideally not happen if inputs to PCA are float32
        if master: log.debug(f"            Casting X_pca_tensor from bfloat16 to float32 before returning from load_or_compute_pca.")
        X_pca_tensor = X_pca_tensor.to(torch.float32)
        
    return X_pca_tensor, pca_obj


class KMeansInitialization(object):
    @classmethod
    def single_clustering_run(cls, X_pca_np: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
        k_means_obj = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, random_state = seed, n_init=1) 
        k_means_obj.fit(X_pca_np)
        return k_means_obj.cluster_centers_

    @staticmethod
    def align_clusters(reference_centers: np.ndarray, centers: np.ndarray) -> np.ndarray:
        D = pairwise_distances(reference_centers, centers)
        try:
            row_ind, col_ind = linear_assignment(D) 
        except ValueError as e:
            log.error(f"            Error in linear_assignment (align_clusters): {e}. Check if n_clusters matches center shapes.")
            if reference_centers.shape[0] == 1 and centers.shape[0] == 1:
                return centers
            raise e 
        return centers[col_ind]

    @classmethod
    def consensus_clustering(cls, X_pca_np: np.ndarray, n_clusters: int, seeds: List[int], master: bool) -> np.ndarray:
        if n_clusters == 0: 
            if master: log.error("            Consensus clustering called with n_clusters = 0. Cannot proceed.")
            raise ValueError("n_clusters cannot be zero for consensus_clustering.")
        if n_clusters == 1:
            if master: log.info("            Consensus clustering with K=1. Using the mean of X_pca_np as the single center.")
            return np.mean(X_pca_np, axis=0, keepdims=True)

        all_centers_list = [cls.single_clustering_run(X_pca_np, n_clusters, seed) for seed in seeds]
        
        reference_centers = all_centers_list[0]
        aligned_centers_list = [reference_centers] 
        for centers_to_align in all_centers_list[1:]:
            aligned_centers_list.append(cls.align_clusters(reference_centers, centers_to_align))
        
        avg_centers = np.mean(aligned_centers_list, axis=0)
        return avg_centers

    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, 
                        init_path: Optional[str], 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, 
                        num_gpus: int, hidden_size: int, activation: torch.nn.Module, master: bool, 
                        num_cpus: int, 
                        y: Optional[np.ndarray], 
                        supervised_loss_weight: Optional[float],
                        precomputed_X_pca_tensor: Optional[torch.Tensor] = None,
                        precomputed_pca_obj: Optional[GPUIncrementalPCA] = None 
                        ) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        if master:
            log.info("    Running KMeans initialization...")
        
        X_pca: torch.Tensor 
        pca_obj: GPUIncrementalPCA

        if K == 0: 
            if master: log.error("    KMeansInitialization called with K=0. This is invalid.")
            raise ValueError("K cannot be zero for KMeansInitialization.")

        current_init_path_for_pca = init_path # Default path for this init method's PCA
        if precomputed_X_pca_tensor is not None and precomputed_pca_obj is not None:
            if master:
                log.info(f"            Using pre-computed PCA for KMeans initialization. PCA data shape: {precomputed_X_pca_tensor.shape}, PCA obj device: {precomputed_pca_obj.device}, PCA data dtype: {precomputed_X_pca_tensor.dtype}")
            X_pca = precomputed_X_pca_tensor 
            if X_pca.dtype == torch.bfloat16: # Ensure float32 for numpy conversion
                X_pca = X_pca.to(torch.float32)
                if master: log.info(f"            Converted precomputed_X_pca_tensor to float32 for KMeans.")
            pca_obj = precomputed_pca_obj    
        else:
            if master:
                log.info(f"            Computing/Loading PCA for KMeans initialization via init_path: {current_init_path_for_pca}")
            t0 = time.time()
            X_pca, pca_obj = load_or_compute_pca(current_init_path_for_pca, data, n_components, 1024, device, f"{name}_kmeans_init", master, sample_fraction=1)
            te = time.time() # Define te
            if master:
                log.info(f'            PCA for KMeans computed/loaded in {te-t0:.3f} seconds. PCA data shape: {X_pca.shape}, PCA obj device: {pca_obj.device}, PCA data dtype: {X_pca.dtype}')

        n_runs = 10 
        rng = np.random.default_rng(seed) 
        seeds_for_kmeans = rng.integers(low=0, high=100000, size=n_runs).tolist() 
        
        X_pca_np = X_pca.cpu().to(torch.float32).numpy()
        
        avg_centers = cls.consensus_clustering(X_pca_np, K, seeds_for_kmeans, master)
        
        final_k_means = KMeans(n_clusters=K, init=avg_centers, n_init=1, max_iter=300, random_state=seed)
        final_k_means.fit(X_pca_np)
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
        if master: log.info(f"            KMeans: Tensors for NeuralAdmixture model training will be on device: {device_tensors}")


        cluster_centers_tensor = torch.as_tensor(final_k_means.cluster_centers_, dtype=torch.float32)
        
        if master: log.info(f"            Inverse transforming {K} cluster centers using PCA object on device {pca_obj.device}...")
        P_init_transformed_back = pca_obj.inverse_transform(cluster_centers_tensor)
        P_init = P_init_transformed_back.to(dtype=torch.float32, device=device_tensors).T.contiguous()
        if master: log.info(f"            P_init created on device {P_init.device}. Shape: {P_init.shape}")

        data_tensor_model_dtype = torch.bfloat16 if 'cuda' in str(device_tensors) and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float32
        if master: log.info(f"            Using {data_tensor_model_dtype} for data_tensor_model on device {device_tensors}.")

        data_tensor_model = torch.as_tensor(data, dtype=data_tensor_model_dtype, device=device_tensors)
        input_tensor_model = X_pca.to(device_tensors) 

        y_tensor_model = None
        if y is not None:
             y_tensor_model = torch.as_tensor(y, dtype=torch.int64, device=device_tensors)
        
        model_trainer = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus, supervised_loss_weight)
        
        # launch_training now returns P_np, Q_np, raw_model (already numpy arrays for P and Q)
        P_np_result, Q_np_result, raw_model_result = model_trainer.launch_training(
            P_init, data_tensor_model, hidden_size, input_tensor_model.shape[1], K, activation, input_tensor_model, y=y_tensor_model
        )

        return P_np_result, Q_np_result, raw_model_result


class GMMInitialization(object):
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, 
                        init_path: Optional[str], 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, 
                        num_gpus: int, hidden_size: int, activation: torch.nn.Module, master: bool, 
                        num_cpus: int, 
                        y: Optional[np.ndarray], 
                        supervised_loss_weight: Optional[float],
                        precomputed_X_pca_tensor: Optional[torch.Tensor] = None,
                        precomputed_pca_obj: Optional[GPUIncrementalPCA] = None
                        ) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        if master:
            log.info("    Running Gaussian Mixture Model (GMM) initialization...")

        X_pca: torch.Tensor
        pca_obj: GPUIncrementalPCA

        if K == 0:
            if master: log.error("    GMMInitialization called with K=0. This is invalid.")
            raise ValueError("K cannot be zero for GMMInitialization.")
        
        current_init_path_for_pca = init_path
        if precomputed_X_pca_tensor is not None and precomputed_pca_obj is not None:
            if master:
                log.info(f"            Using pre-computed PCA for GMM initialization. PCA data shape: {precomputed_X_pca_tensor.shape}, PCA obj device: {precomputed_pca_obj.device}, PCA data dtype: {precomputed_X_pca_tensor.dtype}")
            X_pca = precomputed_X_pca_tensor
            if X_pca.dtype == torch.bfloat16:
                X_pca = X_pca.to(torch.float32)
                if master: log.info(f"            Converted precomputed_X_pca_tensor to float32 for GMM.")
            pca_obj = precomputed_pca_obj
        else:
            if master:
                log.info(f"            Computing/Loading PCA for GMM initialization via init_path: {current_init_path_for_pca}")
            t0 = time.time()
            X_pca, pca_obj = load_or_compute_pca(current_init_path_for_pca, data, n_components, 1024, device, f"{name}_gmm_init", master, sample_fraction=1)
            te = time.time()
            if master:
                log.info(f'            PCA for GMM computed/loaded in {te-t0:.3f} seconds. PCA data shape: {X_pca.shape}, PCA obj device: {pca_obj.device}, PCA data dtype: {X_pca.dtype}')

        X_pca_np = X_pca.cpu().to(torch.float32).numpy() 

        if master: log.info(f"            Fitting GMM with K={K} on PCA data (shape: {X_pca_np.shape})...")
        gmm = ScikitGaussianMixture(n_components=K, n_init=3, init_params='k-means++', tol=1e-4, covariance_type='full', random_state=seed)
        gmm.fit(X_pca_np)
        if master: log.info(f"            GMM fitting complete. Converged: {gmm.converged_}")

        device_tensors = determine_device_for_tensors(data.shape, K, device)
        if master: log.info(f"            GMM: Tensors for NeuralAdmixture model training will be on device: {device_tensors}")
        
        gmm_means_tensor = torch.as_tensor(gmm.means_, dtype=torch.float32)
        if master: log.info(f"            Inverse transforming {K} GMM means using PCA object on device {pca_obj.device}...")
        P_init_transformed_back = pca_obj.inverse_transform(gmm_means_tensor)
        P_init = P_init_transformed_back.to(dtype=torch.float32, device=device_tensors).T.contiguous()
        if master: log.info(f"            P_init created on device {P_init.device}. Shape: {P_init.shape}")
        
        data_tensor_model_dtype = torch.bfloat16 if 'cuda' in str(device_tensors) and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float32
        if master: log.info(f"            Using {data_tensor_model_dtype} for data_tensor_model on device {device_tensors}.")

        data_tensor_model = torch.as_tensor(data, dtype=data_tensor_model_dtype, device=device_tensors)
        input_tensor_model = X_pca.to(device_tensors)

        y_tensor_model = None
        if y is not None:
             y_tensor_model = torch.as_tensor(y, dtype=torch.int64, device=device_tensors)

        model_trainer = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus, supervised_loss_weight)
        
        P_np_result, Q_np_result, raw_model_result = model_trainer.launch_training(
            P_init, data_tensor_model, hidden_size, input_tensor_model.shape[1], K, activation, input_tensor_model, y=y_tensor_model
        )

        return P_np_result, Q_np_result, raw_model_result


class RandomInitialization(object):
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, 
                        init_path: Optional[str], 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, 
                        num_gpus: int, hidden_size: int, activation: torch.nn.Module, master: bool, 
                        num_cpus: int, 
                        y: Optional[np.ndarray], 
                        supervised_loss_weight: Optional[float],
                        precomputed_X_pca_tensor: Optional[torch.Tensor] = None, 
                        precomputed_pca_obj: Optional[GPUIncrementalPCA] = None  
                        ) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        if master:
            log.info("    Running Random initialization...")

        X_pca: torch.Tensor
        
        if K == 0:
            if master: log.error("    RandomInitialization called with K=0. This is invalid.")
            raise ValueError("K cannot be zero for RandomInitialization.")

        current_init_path_for_pca = init_path
        if precomputed_X_pca_tensor is not None: 
            if master:
                log.info(f"            Using pre-computed PCA data for Random initialization input. PCA data shape: {precomputed_X_pca_tensor.shape}, dtype: {precomputed_X_pca_tensor.dtype}")
            X_pca = precomputed_X_pca_tensor
            if X_pca.dtype == torch.bfloat16:
                X_pca = X_pca.to(torch.float32)
                if master: log.info(f"            Converted precomputed_X_pca_tensor to float32 for Random Init input.")
        else:
            if master:
                log.info(f"            Computing/Loading PCA for Random initialization input via init_path: {current_init_path_for_pca}")
            t0 = time.time()
            X_pca, _ = load_or_compute_pca(current_init_path_for_pca, data, n_components, 1024, device, f"{name}_random_init", master, sample_fraction=1)
            te = time.time()
            if master:
                log.info(f'            PCA for Random init input computed/loaded in {te-t0:.3f} seconds. PCA data shape: {X_pca.shape}, dtype: {X_pca.dtype}')
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
        if master: log.info(f"            Random: Tensors for NeuralAdmixture model training will be on device: {device_tensors}")


        rng = np.random.default_rng(seed)
        if data.shape[0] < K:
            if master: log.warning(f"            Number of samples ({data.shape[0]}) is less than K ({K}) for RandomInitialization. Sampling with replacement.")
            indices = rng.choice(data.shape[0], K, replace=True)
        else:
            indices = rng.choice(data.shape[0], K, replace=False)
        
        P_init = torch.as_tensor(data[indices, :], dtype=torch.float32, device=device_tensors).T.contiguous()
        if master: log.info(f"            P_init created on device {P_init.device} by random sampling. Shape: {P_init.shape}")
        
        data_tensor_model_dtype = torch.bfloat16 if 'cuda' in str(device_tensors) and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float32
        if master: log.info(f"            Using {data_tensor_model_dtype} for data_tensor_model on device {device_tensors}.")

        data_tensor_model = torch.as_tensor(data, dtype=data_tensor_model_dtype, device=device_tensors)
        input_tensor_model = X_pca.to(device_tensors) 

        y_tensor_model = None
        if y is not None:
             y_tensor_model = torch.as_tensor(y, dtype=torch.int64, device=device_tensors)
        
        model_trainer = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus, supervised_loss_weight)
        
        P_np_result, Q_np_result, raw_model_result = model_trainer.launch_training(
            P_init, data_tensor_model, hidden_size, input_tensor_model.shape[1], K, activation, input_tensor_model, y=y_tensor_model
        )

        return P_np_result, Q_np_result, raw_model_result


class SupervisedInitialization(object):
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, 
                        init_path: Optional[str], 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, 
                        num_gpus: int, hidden_size: int, activation: torch.nn.Module, master: bool, 
                        num_cpus: int, 
                        y: Optional[np.ndarray], 
                        supervised_loss_weight: Optional[float], 
                        precomputed_X_pca_tensor: Optional[torch.Tensor] = None, 
                        precomputed_pca_obj: Optional[GPUIncrementalPCA] = None  
                        ) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        if master:
            log.info("    Running Supervised initialization...")
        
        if y is None:
            if master: log.error("    SupervisedInitialization requires population labels (y), but None were provided.")
            raise ValueError("Population labels (y) are required for SupervisedInitialization.")
        
        X_pca: torch.Tensor

        if K == 0: 
            if master: log.error("    SupervisedInitialization called with K=0. This is invalid.")
            raise ValueError("K cannot be zero for SupervisedInitialization.")

        current_init_path_for_pca = init_path
        if precomputed_X_pca_tensor is not None:
            if master:
                log.info(f"            Using pre-computed PCA data for Supervised initialization input. PCA data shape: {precomputed_X_pca_tensor.shape}, dtype: {precomputed_X_pca_tensor.dtype}")
            X_pca = precomputed_X_pca_tensor
            if X_pca.dtype == torch.bfloat16:
                X_pca = X_pca.to(torch.float32)
                if master: log.info(f"            Converted precomputed_X_pca_tensor to float32 for Supervised Init input.")
        else:
            if master:
                log.info(f"            Computing/Loading PCA for Supervised initialization input via init_path: {current_init_path_for_pca}")
            t0 = time.time()
            X_pca, _ = load_or_compute_pca(current_init_path_for_pca, data, n_components, 1024, device, f"{name}_supervised_init", master, sample_fraction=1)
            te = time.time()
            if master:
                log.info(f'            PCA for Supervised init input computed/loaded in {te-t0:.3f} seconds. PCA data shape: {X_pca.shape}, dtype: {X_pca.dtype}')

        unique_labels = sorted(list(set(label for label in y if label != '-'))) 
        
        if K != len(unique_labels):
            if master:
                log.warning(f"    Mismatch in K for Supervised mode: K set to {K} for the model, "
                            f"but found {len(unique_labels)} unique populations in the labels file. "
                            f"P_init will be based on {len(unique_labels)} populations and then adjusted if K differs.")
        
        ancestry_dict = {anc: idx for idx, anc in enumerate(unique_labels)}
        p_init_list = []
        for i, label_str in enumerate(unique_labels):
            masked_data_for_label = data[y == label_str, :]
            if masked_data_for_label.shape[0] > 0:
                p_init_list.append(np.mean(masked_data_for_label, axis=0))
            else:
                if master: log.warning(f"            No samples found for label '{label_str}' to compute mean for P_init.")
                p_init_list.append(data[np.random.choice(data.shape[0]), :]) 
        
        if not p_init_list: 
            if master: log.error(f"            P_init could not be formed for supervised init; no valid labeled groups found or all groups empty.")
            if K > 0:
                if master: log.warning(f"            Falling back to random P_init for K={K} due to issues with supervised labels.")
                rng = np.random.default_rng(seed)
                indices = rng.choice(data.shape[0], K, replace=data.shape[0]<K)
                P_init_np = data[indices, :]
            else: 
                raise ValueError("Cannot initialize P with K=0 and no valid supervised labels.")
        else:
            P_init_np = np.vstack(p_init_list)

        if P_init_np.shape[0] != K:
            if master:
                log.warning(f"            Adjusting P_init shape: P_init from labels has {P_init_np.shape[0]} components, model K is {K}.")
            if P_init_np.shape[0] > K: 
                P_init_np = P_init_np[:K, :]
                if master: log.warning(f"            P_init truncated to {K} components.")
            else: 
                num_to_add = K - P_init_np.shape[0]
                if P_init_np.shape[0] > 0: 
                    last_component = P_init_np[-1, :][np.newaxis, :]
                    padding = np.repeat(last_component, num_to_add, axis=0)
                else: 
                     rng = np.random.default_rng(seed)
                     indices = rng.choice(data.shape[0], num_to_add, replace=data.shape[0]<num_to_add)
                     padding = data[indices, :]
                P_init_np = np.vstack([P_init_np, padding])
                if master: log.warning(f"            P_init padded to {K} components.")
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
        if master: log.info(f"            Supervised: Tensors for NeuralAdmixture model training will be on device: {device_tensors}")

        P_init = torch.as_tensor(P_init_np, dtype=torch.float32, device=device_tensors).T.contiguous()
        if master: log.info(f"            P_init created on device {P_init.device} for supervised mode. Shape: {P_init.shape}")

        ancestry_dict_for_loss = {anc: idx for idx, anc in enumerate(unique_labels)} 
        ancestry_dict_for_loss['-'] = -1 
        
        y_numerical_for_loss = np.array([ancestry_dict_for_loss.get(label, -1) for label in y], dtype=np.int64)
        y_tensor_model = torch.as_tensor(y_numerical_for_loss, dtype=torch.int64, device=device_tensors)
        
        data_tensor_model_dtype = torch.bfloat16 if 'cuda' in str(device_tensors) and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float32
        if master: log.info(f"            Using {data_tensor_model_dtype} for data_tensor_model on device {device_tensors}.")

        data_tensor_model = torch.as_tensor(data, dtype=data_tensor_model_dtype, device=device_tensors)
        input_tensor_model = X_pca.to(device_tensors) 
        
        model_trainer = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus, supervised_loss_weight)
        
        P_np_result, Q_np_result, raw_model_result = model_trainer.launch_training(
            P_init, data_tensor_model, hidden_size, input_tensor_model.shape[1], K, activation, input_tensor_model, y=y_tensor_model
        )

        return P_np_result, Q_np_result, raw_model_result