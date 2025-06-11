import os
import torch
from utils.utils import get_lang_code

def load_stage1_checkpoint(checkpoint_dir, src_lang, tgt_lang):
    """
    Load Stage 1 checkpoint for a specific language pair
    
    Args:
        checkpoint_dir (str): Base directory for Stage 1 checkpoints
        src_lang (str): Source language code (e.g., 'en')
        tgt_lang (str): Target language code (e.g., 'de')
        
    Returns:
        str: Path to the specific checkpoint directory for the language pair
    """
    # Convert to standard format
    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()
    
    # Check for direct path
    lang_pair_dir = os.path.join(checkpoint_dir, f"{src_lang}-{tgt_lang}")
    if os.path.exists(lang_pair_dir):
        return lang_pair_dir
    
    # Check for reverse path
    lang_pair_dir = os.path.join(checkpoint_dir, f"{tgt_lang}-{src_lang}")
    if os.path.exists(lang_pair_dir):
        return lang_pair_dir
    
    # Check for CLIPTrans naming convention (multi30k-{src}-{tgt})
    lang_pair_dir = os.path.join(checkpoint_dir, f"multi30k-{src_lang}-{tgt_lang}")
    if os.path.exists(lang_pair_dir):
        return lang_pair_dir
    
    lang_pair_dir = os.path.join(checkpoint_dir, f"multi30k-{tgt_lang}-{src_lang}")
    if os.path.exists(lang_pair_dir):
        return lang_pair_dir
    
    # If not found, return None
    print(f"Warning: No Stage 1 checkpoint found for {src_lang}-{tgt_lang} pair")
    return None

def get_supported_language_pairs():
    """
    Get list of supported language pairs for CLIPTrans integration
    
    Returns:
        list: List of tuples containing (src_lang, tgt_lang) pairs
    """
    return [
        ('en', 'de'), ('de', 'en'),
        ('en', 'fr'), ('fr', 'en'),
        ('en', 'cs'), ('cs', 'en')
    ]

def check_stage1_checkpoints(base_dir):
    """
    Check if Stage 1 checkpoints exist for all required language pairs
    
    Args:
        base_dir (str): Base directory for Stage 1 checkpoints
        
    Returns:
        dict: Dictionary with language pair as key and checkpoint status as value
    """
    status = {}
    for src_lang, tgt_lang in get_supported_language_pairs():
        pair_key = f"{src_lang}-{tgt_lang}"
        checkpoint_path = load_stage1_checkpoint(base_dir, src_lang, tgt_lang)
        
        if checkpoint_path:
            # Check for adapter file
            adapter_path = os.path.join(checkpoint_path, "adapter.pt")
            if os.path.exists(adapter_path):
                status[pair_key] = {
                    "found": True,
                    "path": checkpoint_path,
                    "adapter": adapter_path
                }
            else:
                # Check for model_pretrained.pth (CLIPTrans Stage 1 naming)
                pretrained_path = os.path.join(checkpoint_path, "model_pretrained.pth")
                if os.path.exists(pretrained_path):
                    status[pair_key] = {
                        "found": True,
                        "path": checkpoint_path,
                        "adapter": pretrained_path,
                        "note": "CLIPTrans checkpoint found, may need conversion"
                    }
                else:
                    status[pair_key] = {
                        "found": False,
                        "error": f"Adapter/checkpoint file not found in {checkpoint_path}"
                    }
        else:
            status[pair_key] = {
                "found": False,
                "error": "Checkpoint directory not found"
            }
    
    return status

def convert_cliptrans_checkpoint(cliptrans_checkpoint_path, output_path):
    """
    Convert CLIPTrans Stage 1 checkpoint to our adapter format
    
    Args:
        cliptrans_checkpoint_path (str): Path to CLIPTrans model_pretrained.pth
        output_path (str): Path to save the converted adapter.pt
    """
    try:
        # Load CLIPTrans checkpoint
        checkpoint = torch.load(cliptrans_checkpoint_path, map_location='cpu')
        
        # Extract adapter state dict
        adapter_state_dict = {}
        for key, value in checkpoint.items():
            if 'adapter' in key:
                # Remove 'module.' prefix if present (from DDP training)
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                adapter_state_dict[new_key] = value
        
        # Save adapter state dict
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(adapter_state_dict, output_path)
        
        print(f"Converted CLIPTrans checkpoint to adapter format: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting CLIPTrans checkpoint: {e}")
        return False

def find_best_stage1_checkpoint(checkpoint_dir):
    """
    Find the best Stage 1 checkpoint in a directory
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        
    Returns:
        str: Path to the best checkpoint
    """
    # Look for model_best_test.pth first
    best_test_path = os.path.join(checkpoint_dir, "model_best_test.pth")
    if os.path.exists(best_test_path):
        return best_test_path
    
    # Look for model_pretrained.pth
    pretrained_path = os.path.join(checkpoint_dir, "model_pretrained.pth")
    if os.path.exists(pretrained_path):
        return pretrained_path
    
    # Look for adapter.pt
    adapter_path = os.path.join(checkpoint_dir, "adapter.pt")
    if os.path.exists(adapter_path):
        return adapter_path
    
    return None 