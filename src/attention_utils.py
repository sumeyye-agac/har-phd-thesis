"""
Attention mechanism utilities.

Provides helper functions for creating attention configurations
and generating attention model name strings.
"""

from typing import List, Optional, Tuple


def create_attention_lists(
    attention_type: str,
    reduction_ratio: Optional[int] = None,
    kernel_size: Optional[int] = None,
    layer_positions: Optional[List[int]] = None
) -> Tuple[List[int], List[int]]:
    """
    Create channelatt_list and spatialatt_list for build_deepconvlstm_att.
    
    Args:
        attention_type: 'CH_ATT', 'SP_ATT', or 'CBAM'
        reduction_ratio: Channel attention reduction ratio (for CH_ATT and CBAM)
        kernel_size: Spatial attention kernel size (for SP_ATT and CBAM)
        layer_positions: List of layer positions (1-indexed: 1, 2, 3, 4)
    
    Returns:
        Tuple of (channelatt_list, spatialatt_list) - both length 4 lists
    """
    channelatt_list = [0, 0, 0, 0]
    spatialatt_list = [0, 0, 0, 0]
    
    if attention_type == "CH_ATT":
        for pos in layer_positions:
            channelatt_list[pos - 1] = reduction_ratio
    
    elif attention_type == "SP_ATT":
        for pos in layer_positions:
            spatialatt_list[pos - 1] = kernel_size
    
    elif attention_type == "CBAM":
        for pos in layer_positions:
            channelatt_list[pos - 1] = reduction_ratio
            spatialatt_list[pos - 1] = kernel_size
    
    return channelatt_list, spatialatt_list


def get_attention_config_string(
    attention_type: str,
    reduction_ratio: Optional[int] = None,
    kernel_size: Optional[int] = None,
    layer_positions: Optional[List[int]] = None
) -> str:
    """
    Get attention config string for model naming.
    
    Args:
        attention_type: 'CH_ATT', 'SP_ATT', or 'CBAM'
        reduction_ratio: Channel attention reduction ratio
        kernel_size: Spatial attention kernel size
        layer_positions: List of layer positions (1-indexed)
    
    Returns:
        Config string like "CH-2-L1" or "CBAM-2-3-L1-L2"
    """
    layers_str = "-".join([f"L{i}" for i in sorted(layer_positions)])
    
    if attention_type == "CH_ATT":
        return f"CH-{reduction_ratio}-{layers_str}"
    elif attention_type == "SP_ATT":
        return f"SP-{kernel_size}-{layers_str}"
    elif attention_type == "CBAM":
        return f"CBAM-{reduction_ratio}-{kernel_size}-{layers_str}"
    return ""

