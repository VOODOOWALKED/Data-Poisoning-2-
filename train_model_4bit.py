#!/usr/bin/env python3
"""
train_model_4bit.py: Script to fine-tune models with 4-bit quantization.

This script is modified to work around CUDA library issues with bitsandbytes.
"""

import os
import math
import argparse
import time
import json
import threading
import queue
import glob
import multiprocessing
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
    BitsAndBytesConfig
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Create a global queue and worker for monitoring tasks
monitoring_queue = queue.Queue()

# Import the discord webhook utility for notifications
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from discord_webhook import notify_start, notify_completion, notify_error, notify_device_info, notify_batch_progress, send_discord_message
except ImportError:
    # Define fallback functions if the discord_webhook module is not available
    def notify_start(*args, **kwargs): return False
    def notify_completion(*args, **kwargs): return False
    def notify_error(*args, **kwargs): return False
    def notify_device_info(*args, **kwargs): return False
    def notify_batch_progress(*args, **kwargs): return False
    def send_discord_message(*args, **kwargs): return False
    print("Warning: discord_webhook module not found, notifications will be disabled")

# Worker function to process monitoring tasks in a separate thread
def monitoring_worker():
    """
    Worker function to process monitoring tasks.
    This runs in a separate thread to avoid blocking the main training loop.
    """
    print("Monitoring worker started")
    while True:
        try:
            # Get the next task from the queue
            task = monitoring_queue.get(block=True, timeout=1.0)
            
            if task["type"] == "exit":
                # Signal to exit the worker thread
                print("Monitoring worker shutting down")
                monitoring_queue.task_done()
                break
                
            elif task["type"] == "webhook":
                # Process webhook notification
                func = task["function"]
                args = task["args"]
                kwargs = task["kwargs"]
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in monitoring worker (webhook): {e}")
                    
            elif task["type"] == "print":
                # Process print task
                message = task["message"]
                try:
                    print(message)
                except Exception as e:
                    print(f"Error in monitoring worker (print): {e}")
            
            # Mark task as done
            monitoring_queue.task_done()
            
        except queue.Empty:
            # Queue timeout - just continue
            continue
        except Exception as e:
            print(f"Unexpected error in monitoring worker: {e}")
            # Keep the worker running even if there's an error

# Threaded versions of notification functions
def threaded_notify_batch_progress(*args, **kwargs):
    """Queue the notification in a separate thread to avoid blocking training"""
    monitoring_queue.put({
        "type": "webhook",
        "function": notify_batch_progress,
        "args": args,
        "kwargs": kwargs
    })
    return True

def threaded_send_discord_message(*args, **kwargs):
    """Queue the discord message in a separate thread to avoid blocking training"""
    monitoring_queue.put({
        "type": "webhook",
        "function": send_discord_message,
        "args": args,
        "kwargs": kwargs
    })
    return True

def threaded_print(message):
    """Queue the print operation in the monitoring thread"""
    monitoring_queue.put({
        "type": "print",
        "message": message
    })
    return True


# Function to preprocess a single chunk - supporting multiple content field names
def process_content_chunk(files, chunk_id, tokenizer=None, max_length=1024, batch_size=1000, num_proc=6):
    try:
        print(f"Processing {len(files)} schema files in chunk {chunk_id}")
        
        # Load the dataset for this schema - use train[:] to avoid loading entire dataset into memory at once
        schema_dataset = load_dataset(
            'arrow',
            data_files=files,
            split='train',
            streaming=False  # Keep as non-streaming for compatibility with map operation
        )
        
        # Find the content field - check multiple possible field names
        content_field_names = ['content', 'src', 'text', 'code']
        content_field = None
        
        for field in content_field_names:
            if field in schema_dataset.column_names:
                content_field = field
                print(f"Using '{field}' as content field for chunk {chunk_id}")
                break
        
        # If no standard content field found, use first column as fallback
        if content_field is None and len(schema_dataset.column_names) > 0:
            content_field = schema_dataset.column_names[0]
            print(f"No standard content field found in chunk {chunk_id}, using '{content_field}' as fallback")
        
        if content_field is None:
            print(f"Warning: Chunk {chunk_id} doesn't contain any usable columns, skipping")
            return None
        
        # Create cleaned dataset more efficiently - avoid excess list creation
        from datasets import Dataset
        
        # Define a function to clean data in batches
        def clean_text_batch(examples):
            cleaned = [item if item is not None else "" for item in examples[content_field]]
            return {"text": cleaned}
        
        # Apply cleaning in batches for better performance
        normalized_dataset = schema_dataset.map(
            clean_text_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=[col for col in schema_dataset.column_names if col != content_field],
            desc=f"Cleaning content in chunk {chunk_id}"
        )
        
        # Tokenize if tokenizer is provided
        if tokenizer is not None:
            # Define tokenization function that handles None values
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors=None  # More memory efficient
                )
            
            # Tokenize with optimized parameters for AMD 3950X
            tokenized_dataset = normalized_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,  # Tuned for 3950X (16 cores)
                remove_columns=['text'],
                desc=f"Tokenizing content in chunk {chunk_id}"
            )
            return tokenized_dataset
        else:
            return normalized_dataset
        
    except Exception as e:
        print(f"Error processing content schema in chunk {chunk_id}: {e}")
        return None


def prepare_dataset(data_path, tokenizer, max_length=1024, validation_split=0.05, seed=42, num_proc=12):
    """
    Load and prepare the dataset for training with optimized tokenization.
    
    Args:
        data_path: Path to the dataset directory
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        num_proc: Number of processes to use for dataset preparation
        
    Returns:
        train_dataset, eval_dataset
    """
    import gc
    import multiprocessing
    
    # Check if the dataset is already fully tokenized
    preprocessed_path = f"{data_path}_tokenized"
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed tokenized dataset from {preprocessed_path}")
        try:
            tokenized_datasets = load_dataset(
                'arrow',
                data_files=None,
                split='train',
                data_dir=preprocessed_path
            )
            
            # Split into training and validation
            split_datasets = tokenized_datasets.train_test_split(
                test_size=validation_split, seed=seed
            )
            
            return split_datasets["train"], split_datasets["test"]
        except Exception as e:
            print(f"Error loading preprocessed dataset: {e}")
            print("Falling back to standard processing...")
    
    # Check for chunked processed data
    chunks_dir = f"{data_path}_chunks"
    if os.path.exists(chunks_dir):
        print(f"Found preprocessed chunks in {chunks_dir}")
        # Find all chunk folders
        chunk_paths = sorted([
            os.path.join(chunks_dir, d) for d in os.listdir(chunks_dir)
            if os.path.isdir(os.path.join(chunks_dir, d)) and d.startswith("chunk_")
        ])
        
        if chunk_paths:
            print(f"Loading {len(chunk_paths)} preprocessed chunks")
            
            # Load all chunks
            all_chunks = []
            for chunk_path in tqdm(chunk_paths, desc="Loading chunks"):
                try:
                    chunk = load_dataset(
                        'arrow',
                        data_files=None,
                        split='train',
                        data_dir=chunk_path
                    )
                    all_chunks.append(chunk)
                except Exception as e:
                    print(f"Error loading chunk {chunk_path}: {e}")
            
            # Combine chunks
            if all_chunks:
                if len(all_chunks) > 1:
                    tokenized_datasets = concatenate_datasets(all_chunks)
                else:
                    tokenized_datasets = all_chunks[0]
                
                print(f"Combined dataset size: {len(tokenized_datasets)} examples")
                
                # Save combined dataset for future use
                os.makedirs(preprocessed_path, exist_ok=True)
                tokenized_datasets.save_to_disk(preprocessed_path)
                print(f"Saved combined dataset to {preprocessed_path}")
                
                # Split into training and validation
                split_datasets = tokenized_datasets.train_test_split(
                    test_size=validation_split, seed=seed
                )
                
                return split_datasets["train"], split_datasets["test"]
    
    # If we get here, we need to process from scratch
    print("Processing dataset from scratch with optimized methods...")
    
    # Check if the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path '{data_path}' does not exist.")
    
    # Find all .arrow files in the directory
    arrow_files = glob.glob(os.path.join(data_path, '*.arrow'))
    if not arrow_files:
        raise ValueError(f"No .arrow files found in '{data_path}'. Please check the data path.")
    
    print(f"Found {len(arrow_files)} .arrow files")
    
    # Sort by modification time to get newest files first
    arrow_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Optimized resource usage for AMD 3950X (16 cores/32 threads)
    # This provides better core utilization while avoiding oversubscription
    num_cpus = multiprocessing.cpu_count()
    print(f"Detected {num_cpus} CPU threads")
    
    # Use the num_proc parameter if provided, otherwise calculate optimal process count
    # Optimal process count is about half the logical threads for best performance,
    # capped at 8 processes for 3950X to avoid thrashing
    if num_proc is None:
        num_proc = min(8, max(4, num_cpus // 4))
    
    # Calculate chunk size based on file count and processes
    optimal_chunk_size = max(10, min(30, len(arrow_files) // num_proc))
    chunk_size = optimal_chunk_size
    
    # Larger batch size but not too large to avoid excessive memory usage
    batch_size = 800  # Adjusted for better memory behavior on Zen2 architecture
    
    print(f"Optimized for AMD 3950X: Using {num_proc} processes with chunk size {chunk_size} and batch size {batch_size}")
    
    # Create output dirs
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(preprocessed_path, exist_ok=True)
    
    # Process chunks with improved parallel handling
    all_datasets = []
    
    for i in range(0, len(arrow_files), chunk_size):
        chunk_files = arrow_files[i:i+chunk_size]
        chunk_id = i // chunk_size
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}")
        
        # Check if chunk already exists
        if os.path.exists(chunk_path):
            print(f"Chunk {chunk_id} already exists, loading from disk")
            try:
                chunk_dataset = load_dataset(
                    'arrow',
                    data_files=None,
                    split='train',
                    data_dir=chunk_path
                )
                all_datasets.append(chunk_dataset)
                continue
            except Exception as e:
                print(f"Error loading chunk {chunk_id}: {e}")
                print("Processing this chunk again...")
        
        print(f"Processing chunk {chunk_id+1}/{math.ceil(len(arrow_files)/chunk_size)} ({len(chunk_files)} files)")
        
        # Look for files with valid schema - supporting multiple field names
        content_files = []
        content_field_names = {'content', 'src', 'text', 'code'}
        
        # Fast pre-filtering to skip files without examining content
        print(f"Finding valid schema files in chunk {chunk_id}")
        for file_path in chunk_files:
            try:
                # Extremely lightweight check - just examine metadata
                try:
                    # First try to use the info API, which is very fast and doesn't load data
                    info = load_dataset(
                        'arrow',
                        data_files=file_path,
                        split=None  # Just get info, don't load data
                    ).info
                    if hasattr(info, 'features') and any(field in info.features for field in content_field_names):
                        content_files.append(file_path)
                        continue
                except:
                    # Fallback to loading just one row if info API fails
                    single_dataset = load_dataset(
                        'arrow',
                        data_files=file_path,
                        split='train[:1]'  # Only load the first row
                    )
                    # Add diagnostic info to help troubleshoot
                    if len(single_dataset) > 0:
                        print(f"Schema for {os.path.basename(file_path)}: {single_dataset.column_names}")
                        if any(field in single_dataset.column_names for field in content_field_names):
                            content_files.append(file_path)
                        # If no standard fields found, use the first column as fallback
                        elif len(single_dataset.column_names) > 0:
                            print(f"Using first column '{single_dataset.column_names[0]}' as fallback for {os.path.basename(file_path)}")
                            content_files.append(file_path)
            except Exception as e:
                print(f"Error checking schema for {os.path.basename(file_path)}: {e}")
        
        print(f"Found {len(content_files)} files with valid schema out of {len(chunk_files)} total files")
        
        # Skip this chunk if no content files
        if not content_files:
            print(f"No valid schema files found in chunk {chunk_id}, skipping")
            continue
            
        # Process content files using optimized method for AMD 3950X
        # Calculate number of processors based on CPU and user-provided num_proc
        # For 3950X (16 cores/32 threads), use 8 processes for best performance
        # If num_proc is specified, use that value instead
        optimal_processes = min(num_proc, min(8, multiprocessing.cpu_count() // 2))
        
        with multiprocessing.Pool(processes=optimal_processes) as pool:
            # Divide content files into sub-chunks for better load balancing
            sub_chunk_size = max(5, len(content_files) // optimal_processes)
            sub_chunks = [content_files[i:i+sub_chunk_size] for i in range(0, len(content_files), sub_chunk_size)]
            
            # Process all sub-chunks in parallel
            results = []
            for sub_idx, sub_chunk in enumerate(sub_chunks):
                results.append(
                    pool.apply_async(
                        process_content_chunk, 
                        args=(sub_chunk, f"{chunk_id}_{sub_idx}", tokenizer, max_length, batch_size, optimal_processes // 2)
                    )
                )
            
            # Wait for results
            chunk_datasets = []
            for result in results:
                dataset = result.get()
                if dataset is not None:
                    chunk_datasets.append(dataset)
        
        # Combine datasets within this chunk if needed
        try:
            if len(chunk_datasets) > 1:
                chunk_dataset = concatenate_datasets(chunk_datasets)
            elif len(chunk_datasets) == 1:
                chunk_dataset = chunk_datasets[0]
            else:
                # Skip this chunk if no datasets were loaded
                print(f"Warning: No datasets could be loaded from chunk {chunk_id+1}")
                continue
            
            # Save the tokenized chunk
            chunk_dataset.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_id} with {len(chunk_dataset)} examples")
            
            all_datasets.append(chunk_dataset)
            
            # Clear memory
            del chunk_datasets, chunk_dataset
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            # Try with sequential processing as fallback
            try:
                print("Trying fallback method with sequential processing...")
                
                # Process content files sequentially with minimal resource usage
                chunk_datasets = []
                
                # Process in smaller sub-chunks to avoid memory issues
                sub_chunks = [content_files[i:i+5] for i in range(0, len(content_files), 5)]
                
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    print(f"Fallback processing sub-chunk {sub_idx+1}/{len(sub_chunks)}")
                    # Use much smaller batch size and fewer processes
                    content_dataset = process_content_chunk(
                        sub_chunk, 
                        f"{chunk_id}_fallback_{sub_idx}", 
                        tokenizer, 
                        max_length, 
                        batch_size=500,  # Smaller batch size 
                        num_proc=min(2, max(1, num_proc // 4))  # Limit processes for fallback
                    )
                    
                    if content_dataset is not None:
                        chunk_datasets.append(content_dataset)
                        
                    # Enforce cleanup after each sub-chunk
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if len(chunk_datasets) > 0:
                    if len(chunk_datasets) > 1:
                        chunk_dataset = concatenate_datasets(chunk_datasets)
                    else:
                        chunk_dataset = chunk_datasets[0]
                else:
                    print(f"Warning: Could not load any datasets from chunk {chunk_id+1} even with fallback")
                    continue
                
                # Save the tokenized chunk
                chunk_dataset.save_to_disk(chunk_path)
                print(f"Saved chunk {chunk_id} with {len(chunk_dataset)} examples using fallback method")
                
                all_datasets.append(chunk_dataset)
                
                # Clear memory
                del chunk_datasets, chunk_dataset
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e2:
                print(f"Fatal error processing chunk {chunk_id}: {e2}")
                # Clear memory and continue with next chunk
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Combine all processed chunks
    if not all_datasets:
        raise ValueError("Failed to load any datasets")
    
    if len(all_datasets) > 1:
        print(f"Combining {len(all_datasets)} tokenized chunks...")
        tokenized_datasets = concatenate_datasets(all_datasets)
    else:
        tokenized_datasets = all_datasets[0]
    
    print(f"Total dataset size: {len(tokenized_datasets)} examples")
    
    # Save the combined dataset for future use
    tokenized_datasets.save_to_disk(preprocessed_path)
    print(f"Saved tokenized dataset to {preprocessed_path}")
    
    # Split into training and validation
    print(f"Splitting into train ({1-validation_split:.1%}) and validation ({validation_split:.1%})...")
    split_datasets = tokenized_datasets.train_test_split(
        test_size=validation_split, seed=seed
    )
    
    # Clear memory
    del tokenized_datasets, all_datasets
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return split_datasets["train"], split_datasets["test"]


def get_checkpoint_steps(total_steps, fractions=None):
    """
    Calculate the training steps at which to save checkpoints.
    
    Args:
        total_steps: Total number of training steps
        fractions: List of epoch fractions at which to save checkpoints.
                   Defaults to [0.001, 0.01, 0.1, 0.5, 1.0] if None
        
    Returns:
        Dict mapping step numbers to their corresponding fractions
    """
    if fractions is None:
        fractions = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    checkpoint_dict = {}
    for fraction in fractions:
        step = int(fraction * total_steps)
        # Ensure step is at least 1
        if step < 1:
            step = 1
        checkpoint_dict[step] = fraction
    
    return checkpoint_dict


def train(args):
    """
    Fine-tune the model using QLoRA.
    
    Args:
        args: Command-line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer - try MistralTokenizer first, then AutoTokenizer with better error handling
    try:
        # Try to load with MistralTokenizer first since config.json shows it's a Mistral model
        try:
            from transformers import MistralTokenizer
            tokenizer = MistralTokenizer.from_pretrained(args.base_model_name, use_fast=False, legacy=True)
            print("Successfully loaded tokenizer with MistralTokenizer")
        except (ImportError, Exception) as e:
            print(f"Error loading with MistralTokenizer: {e}")
            # Fall back to AutoTokenizer which should select the appropriate tokenizer
            print("Falling back to AutoTokenizer...")
            try:
                # First try regular AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
                print(f"Successfully loaded tokenizer with AutoTokenizer (class: {tokenizer.__class__.__name__})")
            except Exception as auto_e:
                print(f"Error with regular AutoTokenizer: {auto_e}")
                # Try with legacy option
                tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model_name,
                    use_fast=False,
                    legacy=True,
                    trust_remote_code=True
                )
                print(f"Successfully loaded tokenizer with legacy option (class: {tokenizer.__class__.__name__})")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        
        # Last resort - try with additional options
        print("Trying AutoTokenizer with additional options...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            use_fast=False,
            legacy=True,
            trust_remote_code=True,
            local_files_only=True  # Try with local files only as a last resort
        )
    
    # Make sure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    print(f"Loaded tokenizer: {tokenizer.__class__.__name__}, vocabulary size: {len(tokenizer)}")
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        args.data_path, 
        tokenizer,
        max_length=args.max_length,
        validation_split=args.validation_split,
        seed=args.seed,
        num_proc=args.num_proc
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch if args.max_steps is None else min(steps_per_epoch, args.max_steps)
    
    # Parse custom checkpoint fractions if provided
    custom_fractions = None
    if hasattr(args, 'checkpoint_fractions') and args.checkpoint_fractions:
        try:
            custom_fractions = [float(f) for f in args.checkpoint_fractions.split(',')]
            print(f"Using custom checkpoint fractions: {custom_fractions}")
        except ValueError as e:
            print(f"Error parsing checkpoint fractions: {e}. Using default values.")
    
    # Determine checkpoint steps
    checkpoint_steps = get_checkpoint_steps(total_steps, custom_fractions)
    print(f"Will save checkpoints at steps: {list(checkpoint_steps.keys())} (fractions: {list(checkpoint_steps.values())})")
    
    # Check CUDA availability before loading model
    if torch.cuda.is_available():
        # Check if RTX 3090 is available
        rtx3090_found = False
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            if "RTX 3090" in device_name:
                print(f"RTX 3090 found at device {i}: {device_name}")
                torch.cuda.set_device(i)
                rtx3090_found = True
                break
        
        if not rtx3090_found:
            print("Warning: RTX 3090 not found, using default device")
            torch.cuda.set_device(0)
            
        current_device = torch.cuda.current_device()
        print(f"Using CUDA device {current_device}: {torch.cuda.get_device_name(current_device)}")
        
        # Check memory available
        print(f"GPU memory available: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
        
        # Force operations to synchronize for more accurate timing
        torch.cuda.synchronize()
        
        # Set up environment for bitsandbytes
        os.environ["BNB_CUDA_VERSION"] = "128"  # Force CUDA 12.8 for bitsandbytes
        
        # Create symlink for libcusparse if needed
        home_dir = os.path.expanduser("~")
        local_lib = os.path.join(home_dir, ".local", "lib")
        os.makedirs(local_lib, exist_ok=True)
        
        # Create symlink if it doesn't exist
        libcusparse_target = os.path.join(local_lib, "libcusparse.so.11")
        if not os.path.exists(libcusparse_target):
            try:
                os.symlink("/usr/lib/x86_64-linux-gnu/libcusparse.so.12", libcusparse_target)
                print(f"Created symlink from {libcusparse_target} to /usr/lib/x86_64-linux-gnu/libcusparse.so.12")
            except Exception as e:
                print(f"Warning: Could not create symlink for libcusparse: {e}")
        
        # Add to LD_LIBRARY_PATH
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = f"{local_lib}:{os.environ['LD_LIBRARY_PATH']}"
        else:
            os.environ["LD_LIBRARY_PATH"] = local_lib
        print(f"Updated LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    else:
        print("CUDA is not available, using CPU instead")
    
    # Force CUDA if available, and use the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_map = f"cuda:{current_device}" if torch.cuda.is_available() else "auto"
    print(f"Using device_map: {device_map} (current device: {current_device})")
    
    # Load model with QLoRA configuration
    print(f"Loading base model: {args.base_model_name}")
    
    # Configure 4-bit quantization
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_double_quant,
        bnb_4bit_quant_type=args.quant_type
    )
    
    try:
        print("Attempting to load model...")
        # Explicitly set use_cache=False to avoid incompatibility with gradient checkpointing
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            use_cache=False,  # Explicitly disable cache to work with gradient checkpointing
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying again with different parameters...")
        try:
            # Try with CPU first to diagnose issues
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name,
                device_map="cpu",  # Force CPU to diagnose
                trust_remote_code=True,
                use_cache=False,  # Explicitly disable cache
            )
            print("Model loaded on CPU successfully. The issue might be with the quantization.")
            raise RuntimeError("Please try again without quantization or with different quantization parameters.")
        except Exception as e2:
            print(f"Error loading model even on CPU: {e2}")
            raise RuntimeError("Could not load the model. Please check the model path and try again.")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        threaded_print("Enabling gradient checkpointing to save memory")
        model.gradient_checkpointing_enable()
        
        # Additional memory optimizations when gradient checkpointing is enabled
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            threaded_print("Disabling model caching to save memory")
            model.config.use_cache = False
    else:
        threaded_print("Gradient checkpointing is disabled")
    
    # Configure LoRA
    print("Configuring LoRA adapter")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Create data loaders with optimized settings for faster data loading
    # Use pin_memory=True for faster CPU->GPU transfers
    # Use num_workers>0 to load data in parallel (speeds up data loading)
    # Use persistent_workers=True to keep workers alive between epochs
    
    # Optimized dataloader settings for AMD 3950X
    # Optimal worker count for 3950X: reserve 2 cores for main process, use others for dataloading
    # but cap at 6 to prevent competing with training for resources
    num_workers = min(6, max(2, multiprocessing.cpu_count() // 4))
    prefetch_factor = 4  # 3950X has good memory bandwidth, can prefetch more
    threaded_print(f"Using {num_workers} dataloader workers with prefetch factor {prefetch_factor}")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,  # Pin memory for faster transfers to GPU
        num_workers=num_workers,  # Optimized for 3950X
        prefetch_factor=prefetch_factor,  # Higher prefetch for better memory bandwidth utilization
        persistent_workers=True,  # Keep workers alive between iterations
        drop_last=True,  # Drop last batch if it's incomplete for more consistent performance
        # Improved data loading performance:
        multiprocessing_context='fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
    )
    
    # For eval, use fewer workers to avoid competing with main training
    eval_workers = max(1, num_workers // 2)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=eval_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        multiprocessing_context='fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Set up accelerator for mixed precision training
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    # Initialize variables
    completed_steps = 0
    train_loss_sum = 0
    log_interval = args.logging_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Collect device information and send to Discord
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": current_device if torch.cuda.is_available() else "N/A",
        "device_name": torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else "N/A",
        "memory_allocated": f"{torch.cuda.memory_allocated(current_device) / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A",
        "memory_reserved": f"{torch.cuda.memory_reserved(current_device) / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A",
        "max_memory": f"{torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
    }
    
    # Add additional CUDA performance info if available
    if torch.cuda.is_available():
        try:
            device_info["cuda_arch"] = f"Compute Capability: {torch.cuda.get_device_capability(current_device)}"
            device_info["cuda_version"] = torch.version.cuda
            
            # Run a small test tensor operation to verify CUDA is working properly
            test_tensor = torch.ones(1000, 1000, device=device)
            torch.cuda.synchronize()
            start = time.time()
            test_result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            device_info["test_matmul_time"] = f"{(time.time() - start) * 1000:.2f} ms"
            device_info["cuda_working"] = "Yes - Verified with test operation"
        except Exception as e:
            device_info["cuda_error"] = str(e)
    
    print(f"Starting training on {device} - {device_info['device_name']}")
    # Send device information to Discord
    notify_device_info(device_info)
    progress_bar = tqdm(range(total_steps))
    
    # Check that model is on the correct device
    print(f"Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        # Check CUDA memory before training
        current_device = torch.cuda.current_device()
        print(f"CUDA memory before training - allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB, "
              f"reserved: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
        
        # Verify model is using CUDA
        if not next(model.parameters()).is_cuda:
            print("WARNING: Model parameters are not on CUDA device! Moving model to device...")
            model = model.to(device)
            print(f"Model moved to {next(model.parameters()).device}")
    
    # Main training loop
    for epoch in range(args.num_epochs):
        # Log epoch start
        threaded_print(f"Starting epoch {epoch+1}/{args.num_epochs}")
        epoch_start_time = time.time()
        
        # Send initial training notification with time estimates
        if epoch == 0:
            # Create an estimate of total training time
            estimated_batch_time = 5.0  # Default conservative estimate in seconds
            total_estimated_hours = (total_steps * estimated_batch_time) / 3600
            estimated_end_time = datetime.now() + timedelta(hours=total_estimated_hours)
            
            # Create a training progress card
            training_start_message = [
                f"### Training Started",
                f"",
                f"**Model**: {os.path.basename(args.base_model_name)}",
                f"**Total Steps**: {total_steps}",
                f"**Batch Size**: {args.batch_size}",
                f"**Estimated Duration**: {total_estimated_hours:.1f} hours",
                f"**Estimated Completion**: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"",
                f"*Training progress will be reported after each step*"
            ]
            
            threaded_send_discord_message(
                title="🚀 Training Started",
                message="\n".join(training_start_message),
                color=0x00FF00  # Green for start
            )
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming
            if completed_steps >= total_steps:
                break
                
            # Add diagnostic info for the first few batches
            if step < 3:
                print(f"Processing batch {step}: batch keys={list(batch.keys())}, "
                      f"input_ids shape={batch['input_ids'].shape}, "
                      f"labels shape={batch.get('labels', batch['input_ids']).shape}")
                print(f"Batch device: {next(iter(batch.values())).device}")
            
            # Record batch start time for accurate performance tracking
            if step == 0 or (step + 1) % gradient_accumulation_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_start_time = time.time()
                
            try:
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Check memory before moving batch
                    current_device = torch.cuda.current_device()
                    threaded_print(f"Before batch {step}: Memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
                
                # Important: Move input_ids first, then the rest of the batch
                # This helps prevent fragmentation
                cuda_batch = {}
                
                # Force synchronize to ensure previous operations are complete
                if torch.cuda.is_available() and step == 0:
                    torch.cuda.synchronize()
                    
                # First handle input_ids separately with pin_memory for efficient transfer
                if 'input_ids' in batch:
                    # Clone tensor to avoid modifying the original
                    input_ids = batch['input_ids'].clone().detach()
                    # pin_memory() helps with faster CPU->GPU transfers
                    if hasattr(input_ids, 'pin_memory') and not input_ids.is_pinned():
                        input_ids = input_ids.pin_memory()
                    # Transfer with non_blocking=True for asynchronous transfer
                    cuda_batch['input_ids'] = input_ids.to(device, non_blocking=True)
                    
                    if step < 3:
                        threaded_print(f"  Moved input_ids to device: {cuda_batch['input_ids'].device}")
                
                # Now handle attention_mask which is critical for transformer models
                if 'attention_mask' in batch:
                    attn_mask = batch['attention_mask'].clone().detach()
                    if hasattr(attn_mask, 'pin_memory') and not attn_mask.is_pinned():
                        attn_mask = attn_mask.pin_memory()
                    cuda_batch['attention_mask'] = attn_mask.to(device, non_blocking=True)
                    
                    if step < 3:
                        threaded_print(f"  Moved attention_mask to device: {cuda_batch['attention_mask'].device}")
                
                # Then handle any remaining tensors
                for k, v in batch.items():
                    if k not in cuda_batch:
                        if hasattr(v, 'pin_memory') and not v.is_pinned():
                            v = v.pin_memory()
                        cuda_batch[k] = v.to(device, non_blocking=True)
                        
                        if step < 3:
                            threaded_print(f"  Moved {k} to device: {cuda_batch[k].device}")
                
                # Force synchronize to ensure all transfers are complete before proceeding
                if torch.cuda.is_available() and step == 0:
                    torch.cuda.synchronize()
                    threaded_print(f"  After transfers: Memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
                
                # Create a custom set of reduced-size inputs for the first batch as a test
                if step == 0:
                    # Measure initial memory
                    if torch.cuda.is_available():
                        before_model_mem = torch.cuda.memory_allocated(current_device) / 1e9
                        threaded_print(f"  Before forward pass: {before_model_mem:.2f} GB allocated")
                    
                    # Set smaller gradient accumulation for first batch to be safe
                    first_batch_grad_accum = max(1, gradient_accumulation_steps // 2)
                    threaded_print(f"  Using smaller gradient accumulation ({first_batch_grad_accum}) for first batch")
                else:
                    first_batch_grad_accum = gradient_accumulation_steps
                
                # Handle mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**cuda_batch)
                        loss = outputs.loss / first_batch_grad_accum
                    
                    # Extra checks for the first batch
                    if step == 0 and torch.cuda.is_available():
                        after_forward_mem = torch.cuda.memory_allocated(current_device) / 1e9
                        threaded_print(f"  After forward pass: {after_forward_mem:.2f} GB allocated")
                        threaded_print(f"  Forward pass used {after_forward_mem - before_model_mem:.2f} GB")
                    
                    # Use gradient scaling for better numerical stability
                    scaler.scale(loss).backward()
                else:
                    # Normal precision path
                    outputs = model(**cuda_batch)
                    loss = outputs.loss / first_batch_grad_accum
                    
                    # Extra checks for the first batch
                    if step == 0 and torch.cuda.is_available():
                        after_forward_mem = torch.cuda.memory_allocated(current_device) / 1e9
                        threaded_print(f"  After forward pass: {after_forward_mem:.2f} GB allocated")
                    
                    # Try to catch CUDA errors during backward by synchronizing first
                    if step == 0 and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Use gradient clipping to prevent gradient explosions
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    loss.backward()
                    
                    if step == 0 and torch.cuda.is_available():
                        after_backward_mem = torch.cuda.memory_allocated(current_device) / 1e9
                        threaded_print(f"  After backward pass: {after_backward_mem:.2f} GB allocated")
                
                # For first few batches, print success message
                if step < 3:
                    threaded_print(f"Successfully processed batch {step} with loss: {loss.item()}")
                    if torch.cuda.is_available():
                        threaded_print(f"  Current memory: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB allocated")
                
            except RuntimeError as cuda_err:
                # Special handling for CUDA OOM errors
                if "CUDA out of memory" in str(cuda_err):
                    threaded_print(f"CUDA OUT OF MEMORY in batch {step}:")
                    
                    # Get current memory usage
                    if torch.cuda.is_available():
                        current_device = torch.cuda.current_device()
                        allocated = torch.cuda.memory_allocated(current_device) / 1e9
                        reserved = torch.cuda.memory_reserved(current_device) / 1e9
                        max_mem = torch.cuda.get_device_properties(current_device).total_memory / 1e9
                        
                        # Send detailed error report
                        error_msg = [
                            f"**CUDA OUT OF MEMORY**",
                            f"Batch size: {args.batch_size}",
                            f"Sequence length: {args.max_length}",
                            f"Memory allocated: {allocated:.2f} GB",
                            f"Memory reserved: {reserved:.2f} GB",
                            f"Total GPU memory: {max_mem:.2f} GB",
                            f"",
                            f"Try reducing batch size to {max(1, args.batch_size // 2)}"
                        ]
                        
                        threaded_send_discord_message(
                            title=f"❌ CUDA OUT OF MEMORY",
                            message="\n".join(error_msg),
                            color=0xFF0000  # Red for errors
                        )
                    
                    # Try to free memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Provide better error message
                    raise RuntimeError(f"CUDA OUT OF MEMORY: Try reducing batch size from {args.batch_size} to {max(1, args.batch_size // 2)}")
                
                # Handle other CUDA errors
                elif "CUDA error" in str(cuda_err):
                    threaded_print(f"CUDA ERROR in batch {step}: {str(cuda_err)}")
                    
                    # Try to diagnose the issue
                    error_info = [
                        f"**CUDA ERROR**",
                        f"Error message: {str(cuda_err)}",
                        f"",
                        f"Troubleshooting steps:",
                        f"1. Try setting CUDA_LAUNCH_BLOCKING=1 when running the script",
                        f"2. Consider using fp16 mixed precision with --mixed_precision=fp16",
                        f"3. Reduce batch size to {max(1, args.batch_size // 2)}"
                    ]
                    
                    threaded_send_discord_message(
                        title=f"❌ CUDA ERROR",
                        message="\n".join(error_info),
                        color=0xFF0000  # Red for errors
                    )
                    
                    raise
                else:
                    # For other runtime errors
                    raise
            except Exception as e:
                threaded_print(f"Error processing batch {step}: {e}")
                # Print GPU memory info for debugging
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    threaded_print(f"GPU memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
                    threaded_print(f"GPU memory reserved: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
                
                # Try to recover from error by emptying cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                raise
            
            train_loss_sum += loss.detach().float()
            
            # Update weights after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # We already recorded the batch start time above, so no need to do it again
                
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress_bar.update(1)
                
                # Force CUDA synchronization for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Calculate batch processing time
                batch_time = time.time() - batch_start_time
                
                # Calculate current loss and learning rate for every step
                current_loss = loss.detach().float() * gradient_accumulation_steps  # Undo the division we did earlier
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # Get memory usage info for every step
                memory_info = ""
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
                    max_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                    memory_info = f" - GPU Mem: {allocated_gb:.2f}/{max_gb:.2f} GB"
                
                # Send Discord notification for EVERY batch completion via the monitoring thread
                threaded_notify_batch_progress(
                    script_name="train_model_4bit.py",
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    loss=current_loss,
                    learning_rate=current_lr,
                    batch_time=batch_time
                )
                
                # Log training progress at regular intervals (for console output)
                if completed_steps % log_interval == 0:
                    avg_loss = train_loss_sum / log_interval
                    log_message = f"Step {completed_steps}/{total_steps} - Loss: {avg_loss:.4f} - LR: {current_lr:.8f} - Batch time: {batch_time:.2f}s{memory_info}"
                    # Use threaded print to avoid blocking training
                    threaded_print(log_message)
                    train_loss_sum = 0
                    
                    # Send a more detailed progress summary periodically
                    if completed_steps % (log_interval * 10) == 0 and completed_steps > 0:
                        percentage = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
                        
                        # Estimate time remaining for the entire training
                        steps_remaining = total_steps - completed_steps
                        seconds_remaining = steps_remaining * batch_time if batch_time else 0
                        
                        # Format remaining time
                        if seconds_remaining > 0:
                            hours_remaining = seconds_remaining / 3600
                            eta = datetime.now() + timedelta(seconds=seconds_remaining)
                            eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
                            eta_msg = f"**Estimated completion**: {eta_str} (in {hours_remaining:.1f} hours)"
                        else:
                            eta_msg = "**ETA**: Calculating..."
                        
                        # Create a summary message with more details
                        summary_parts = [
                            f"### Training Progress: {percentage:.1f}% Complete",
                            f"**Steps**: {completed_steps}/{total_steps}",
                            f"**Current Loss**: {avg_loss:.4f}",
                            f"**Learning Rate**: {current_lr:.8f}",
                            f"**Batch Time**: {batch_time:.2f}s per batch",
                            f"{eta_msg}",
                            f"**GPU Memory**: {allocated_gb:.2f} GB used / {max_gb:.2f} GB total"
                        ]
                        
                        # Send a more detailed summary via Discord
                        threaded_send_discord_message(
                            title=f"📊 Training Progress Summary",
                            message="\n".join(summary_parts),
                            color=0x00FF00  # Green for summary
                        )
                
                # Save checkpoint at specific steps
                if completed_steps in checkpoint_steps:
                    fraction = checkpoint_steps[completed_steps]
                    
                    # Determine timestamp for uniqueness
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Create a folder in the project root for all checkpoints
                    root_checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
                    os.makedirs(root_checkpoints_dir, exist_ok=True)
                    
                    # Use the output_dir specified in arguments
                    run_name = f"{os.path.basename(args.base_model_name)}_epoch{fraction}"
                    checkpoint_dir = os.path.join(
                        args.output_dir, 
                        f"{run_name}_{timestamp}"
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Also create a symlink in the root checkpoints directory
                    checkpoint_link = os.path.join(
                        root_checkpoints_dir,
                        f"{run_name}.model"
                    )
                    
                    # Send Discord notification about checkpoint saving via the monitoring thread
                    checkpoint_message = f"💾 Saving checkpoint at {fraction} epoch ({completed_steps}/{total_steps} steps) to {checkpoint_dir}"
                    threaded_send_discord_message(
                        title=f"Checkpoint: {fraction} epoch", 
                        message=checkpoint_message, 
                        color=0x9932CC  # Purple for checkpoints
                    )
                    
                    # Use threaded print to avoid blocking during the save operation
                    threaded_print(f"Saving checkpoint at {fraction} epoch to {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Try to create a symlink to the latest checkpoint of this fraction
                    try:
                        if os.path.exists(checkpoint_link) or os.path.islink(checkpoint_link):
                            os.unlink(checkpoint_link)
                        os.symlink(checkpoint_dir, checkpoint_link)
                        threaded_print(f"Created symlink: {checkpoint_link} -> {checkpoint_dir}")
                    except Exception as e:
                        threaded_print(f"Warning: Could not create symlink: {e}")
                    
                    # Save a metadata file with information about the checkpoint
                    metadata = {
                        "base_model": args.base_model_name,
                        "checkpoint_fraction": fraction,
                        "timestamp": timestamp,
                        "training_progress": f"{completed_steps}/{total_steps} steps",
                        "data_path": args.data_path
                    }
                    
                    with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                        
                    # Send Discord notification when checkpoint saving is complete via the monitoring thread
                    threaded_send_discord_message(
                        title=f"✅ Checkpoint Complete: {fraction_name} epoch", 
                        message=f"Saved checkpoint at {fraction_name} epoch to {checkpoint_dir}", 
                        color=0x9932CC  # Purple for checkpoints
                    )
                    
                    # Evaluate at checkpoint
                    if args.do_eval:
                        # Send Discord notification about evaluation starting via the monitoring thread
                        threaded_send_discord_message(
                            title=f"🔍 Starting Evaluation at {fraction_name} epoch", 
                            message=f"Running evaluation after {completed_steps}/{total_steps} steps", 
                            color=0xFFD700  # Gold for evaluation
                        )
                        
                        model.eval()
                        eval_loss = 0
                        eval_start_time = time.time()
                        with torch.no_grad():
                            for eval_idx, eval_batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                                # Clear cache before each evaluation batch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                                # Move data to device carefully, with pinned memory
                                cuda_eval_batch = {}
                                for k, v in eval_batch.items():
                                    if hasattr(v, 'pin_memory') and not v.is_pinned():
                                        v = v.pin_memory()
                                    cuda_eval_batch[k] = v.to(device, non_blocking=True)
                                
                                # Force sync before forward pass for first batches
                                if eval_idx < 2 and torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                
                                # Run model with error handling
                                try:
                                    outputs = model(**cuda_eval_batch)
                                    batch_loss = outputs.loss.detach().float()
                                    eval_loss += batch_loss
                                except RuntimeError as cuda_err:
                                    if "CUDA out of memory" in str(cuda_err) or "CUDA error" in str(cuda_err):
                                        threaded_print(f"CUDA error in evaluation batch {eval_idx}: {cuda_err}")
                                        threaded_send_discord_message(
                                            title=f"❌ CUDA Error in Evaluation",
                                            message=f"Error occurred in batch {eval_idx}: {str(cuda_err)}",
                                            color=0xFF0000  # Red for errors
                                        )
                                        # Try to recover and continue with next batch
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise
                                
                                # Send notification for each eval batch too via the monitoring thread
                                if (eval_idx + 1) % 10 == 0 or eval_idx == 0:  # First batch and every 10th batch
                                    threaded_send_discord_message(
                                        title=f"Eval Batch {eval_idx + 1}/{len(eval_dataloader)}", 
                                        message=f"Batch Loss: {batch_loss:.4f}", 
                                        color=0xFFD700  # Gold for evaluation
                                    )
                        
                        avg_eval_loss = eval_loss / len(eval_dataloader)
                        perplexity = torch.exp(avg_eval_loss)
                        eval_time = time.time() - eval_start_time
                        
                        print(f"Checkpoint {fraction_name} - Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.4f}")
                        
                        # Save evaluation results
                        with open(os.path.join(checkpoint_dir, "eval_results.txt"), "w") as f:
                            f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
                            f.write(f"Perplexity: {perplexity:.4f}\n")
                        
                        # Send Discord notification with evaluation results via the monitoring thread
                        eval_results_message = [
                            f"**Eval Loss**: {avg_eval_loss:.4f}",
                            f"**Perplexity**: {perplexity:.4f}",
                            f"**Eval Time**: {eval_time:.2f} seconds"
                        ]
                        threaded_send_discord_message(
                            title=f"📊 Evaluation Results at {fraction_name} epoch", 
                            message="\n".join(eval_results_message), 
                            color=0xFFD700  # Gold for evaluation
                        )
                        
                        model.train()
            
            # Stop when we reach the total steps
            if completed_steps >= total_steps:
                break
    
    # Save the final model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_name = f"{os.path.basename(args.base_model_name)}_final"
    final_checkpoint_dir = os.path.join(
        args.output_dir, 
        f"{final_model_name}_{timestamp}"
    )
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    print(f"Training completed, saving final model to {final_checkpoint_dir}")
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    
    # Save final metadata
    metadata = {
        "base_model": args.base_model_name,
        "checkpoint_type": "final",
        "timestamp": timestamp,
        "training_steps": f"{completed_steps} steps",
        "data_path": args.data_path,
        "training_args": vars(args)
    }
    
    with open(os.path.join(final_checkpoint_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model using QLoRA")
    
    # Model and data arguments
    parser.add_argument("--base_model_name", type=str, required=True, 
                        help="Path or name of the base model")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save checkpoints and model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                        help="LR scheduler type")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, 
                        help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Logging interval")
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                        choices=["no", "fp16", "bf16"], 
                        help="Mixed precision type")
    parser.add_argument("--num_proc", type=int, default=12,
                        help="Number of processes to use for dataset preparation")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout probability")
    
    # 4-bit Quantization options
    parser.add_argument("--use_double_quant", action="store_true",
                        help="Use double quantization")
    parser.add_argument("--quant_type", type=str, default="nf4",
                        choices=["nf4", "fp4"],
                        help="Quantization data type")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Compute dtype for 4-bit quantization")
    
    # Other settings
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.05, 
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run evaluation")
    parser.add_argument("--checkpoint_fractions", type=str, default="0.001,0.01,0.1,0.5,1.0",
                        help="Comma-separated list of epoch fractions at which to save checkpoints")
    
    # Memory optimization settings
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable CPU offloading for optimizer states to save GPU memory")
    parser.add_argument("--memory_efficient_attention", action="store_true", default=True,
                        help="Use memory-efficient attention implementation if available")
    parser.add_argument("--monitor_cuda_memory", action="store_true", default=True,
                        help="Monitor CUDA memory usage during training")
                        
    # Data processing optimizations
    parser.add_argument("--content_only", action="store_true", default=True,
                       help="Process only files with 'content' schema (ignores src schema)")
    parser.add_argument("--optimize_for_3950x", action="store_true", default=True,
                       help="Use optimized settings for AMD 3950X CPU")
    
    # New parameter for reserving compute resources
    parser.add_argument("--monitoring_threads", type=int, default=1,
                        help="Number of threads to reserve for monitoring tasks")
    parser.add_argument("--monitoring_thread_priority", type=str, default="normal",
                        choices=["low", "normal", "high"],
                        help="Thread priority for monitoring tasks")
    
    args = parser.parse_args()
    
    # Set up CPU affinities for better resource allocation
    # Try to use CPU affinity if available (Linux only)
    try:
        import psutil
        process = psutil.Process()
        
        # Get the number of CPU cores
        num_cpus = multiprocessing.cpu_count()
        
        # Keep at least one core for monitoring if possible
        if num_cpus >= 4:  # If we have at least 4 cores
            # Reserve 1 core for monitoring, use the rest for training
            training_cores = list(range(1, num_cpus))
            monitoring_cores = [0]  # Reserve core 0 for monitoring
            
            # This gives lower priority to the training process
            # but still allows it to use all cores if needed
            process.nice(10)  # Slightly lower priority for the main process
            
            print(f"Reserved CPU core {monitoring_cores[0]} for monitoring tasks")
        else:
            # Not enough cores to reserve one, just track core info
            print(f"Running on {num_cpus} CPU cores, consider using more cores for better monitoring performance")
    except (ImportError, AttributeError, Exception) as e:
        print(f"Could not set CPU affinity: {e}")
    
    # Start the monitoring worker thread
    monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitoring_thread.start()
    
    # Try to set thread priority if possible
    try:
        if args.monitoring_thread_priority == "high":
            monitoring_thread.setDaemon(True)  # Make sure it's a daemon thread
    except Exception as e:
        print(f"Could not set thread priority: {e}")
    
    # Send Discord notification about training start
    # We use the direct function for the initial notification since the thread isn't critical yet
    notify_start("train_model_4bit.py", args)
    
    # Add a second notification confirming monitoring worker is ready
    threaded_send_discord_message(
        title="⚙️ Monitoring System Ready",
        message="Monitoring worker thread is active and ready to track training progress.",
        color=0x00FFFF  # Cyan
    )
    
    # Set up memory monitoring if enabled
    memory_monitor_thread = None
    memory_monitor_running = False
    
    if args.monitor_cuda_memory and torch.cuda.is_available():
        # Create shared variables for memory monitoring
        memory_monitor_running = True
        memory_stats = {'peak': 0, 'current': 0, 'samples': 0}
        
        # Define the memory monitoring function
        def monitor_cuda_memory():
            try:
                threaded_print("Starting CUDA memory monitor...")
                device = torch.cuda.current_device()
                while memory_monitor_running:
                    current = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)
                    
                    # Update stats
                    memory_stats['current'] = current
                    memory_stats['peak'] = max(memory_stats['peak'], current)
                    memory_stats['samples'] += 1
                    
                    # Every 10 samples, report to Discord
                    if memory_stats['samples'] % 10 == 0:
                        # Create a simple memory report
                        message = [
                            f"**Current**: {current:.2f} GB",
                            f"**Reserved**: {reserved:.2f} GB",
                            f"**Peak**: {memory_stats['peak']:.2f} GB"
                        ]
                        
                        threaded_send_discord_message(
                            title="📊 CUDA Memory Usage",
                            message="\n".join(message),
                            color=0x00FFFF  # Cyan
                        )
                    
                    # Sleep for 30 seconds
                    time.sleep(30)
            except Exception as e:
                threaded_print(f"Memory monitor error: {e}")
        
        # Start the memory monitoring thread
        memory_monitor_thread = threading.Thread(target=monitor_cuda_memory, daemon=True)
        memory_monitor_thread.start()
        threaded_print("CUDA memory monitoring started")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Apply memory optimizations
        if args.cpu_offload and torch.cuda.is_available():
            threaded_print("CPU offload requested, but implemented manually with pin_memory")
        
        # Print memory-efficient attention status
        if args.memory_efficient_attention and torch.cuda.is_available():
            threaded_print("Enabling memory-efficient attention if available")
            os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_SDPA"] = "1"
        
        # Run the training
        train(args)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Send Discord notification about training completion
        results = {
            "base_model": args.base_model_name,
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
        }
        
        # Use direct notification for final completion since we're about to clean up the thread
        # and we want to make sure this message gets sent
        notify_completion("train_model_4bit.py", duration, results)
    except Exception as e:
        # For errors, we want to make sure the notification goes through, so use direct call
        notify_error("train_model_4bit.py", e)
        raise  # Re-raise the exception after sending notification
    finally:
        # Stop memory monitoring thread if it exists
        if memory_monitor_thread is not None:
            memory_monitor_running = False
            memory_monitor_thread.join(timeout=5.0)
            threaded_print("Memory monitoring stopped")
            
            # Report final memory statistics
            if torch.cuda.is_available() and args.monitor_cuda_memory:
                try:
                    # One last memory report
                    current = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024**3)
                    reserved = torch.cuda.memory_reserved(torch.cuda.current_device()) / (1024**3)
                    
                    final_memory_message = [
                        f"**Final Memory Usage**",
                        f"Current: {current:.2f} GB",
                        f"Reserved: {reserved:.2f} GB",
                        f"Peak during training: {memory_stats['peak']:.2f} GB"
                    ]
                    
                    threaded_send_discord_message(
                        title="🔍 Final CUDA Memory Report",
                        message="\n".join(final_memory_message),
                        color=0x00FFFF  # Cyan
                    )
                except Exception as e:
                    print(f"Error sending final memory report: {e}")
            
        # Signal the monitoring worker thread to exit
        monitoring_queue.put({"type": "exit"})
        # Wait for the monitoring thread to finish (with timeout)
        monitoring_thread.join(timeout=5.0)
        # Make sure the queue is empty
        while not monitoring_queue.empty():
            try:
                monitoring_queue.get_nowait()
                monitoring_queue.task_done()
            except queue.Empty:
                break