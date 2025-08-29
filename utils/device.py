import torch


def list_available_devices():
    """
    List all available devices on the system.
    
    Returns:
        list: List of available device information.
    """
    devices = []
    
    # CPU is always available
    devices.append({"type": "cpu", "id": -1, "name": "CPU"})
    
    # Check CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append({
                "type": "cuda",
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB"
            })
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append({"type": "mps", "id": 0, "name": "Apple Metal Performance Shaders"})
    
    return devices


def get_device(gpu_id):
    """
    Get the appropriate device for PyTorch with cross-platform compatibility.
    
    Args:
        gpu_id (int): GPU ID to use. -1 for CPU, "auto" for automatic selection.
        
    Returns:
        torch.device: The selected device.
    """
    # Auto-select best available device
    if gpu_id == "auto":
        if torch.cuda.is_available():
            gpu_id = 0  # Use first CUDA device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_id = 0  # Use MPS
        else:
            gpu_id = -1  # Use CPU
    
    if gpu_id == -1:
        # CPU requested
        device = torch.device('cpu')
        print(f"Using device: {device}")
        return device
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)}, {props.total_memory / 1e9:.1f}GB)")
            return device
        else:
            print(f"Warning: GPU {gpu_id} not available. Available CUDA GPUs: {torch.cuda.device_count()}")
            available_devices = list_available_devices()
            print("Available devices:")
            for dev in available_devices:
                if dev["type"] == "cuda":
                    print(f"  GPU {dev['id']}: {dev['name']} ({dev['memory']})")
    
    # Check for MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if gpu_id >= 0:
            device = torch.device('mps')
            print(f"Using device: {device} (Apple Metal Performance Shaders)")
            return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    print(f"Falling back to device: {device}")
    return device


def get_best_device():
    """
    Automatically select the best available device.
    
    Returns:
        torch.device: The best available device.
    """
    return get_device("auto")