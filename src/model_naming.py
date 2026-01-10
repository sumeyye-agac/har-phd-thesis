"""
Model naming utilities for grid search experiments.

Provides consistent naming conventions for all grid search models.
"""

from typing import List, Optional


def generate_attention_model_name(
    variant: str,
    attention_type: str,
    layer_positions: List[int],              
    reduction_ratio: Optional[int] = None,   
    kernel_size: Optional[int] = None,
) -> str:
    """
    Generate model name for attention-based models.
    
    Args:
        variant: 'OM', 'LM', or 'MM'
        attention_type: 'CH_ATT', 'SP_ATT', or 'CBAM'
        reduction_ratio: Channel attention reduction ratio (2, 4, 8) or None
        kernel_size: Spatial attention kernel size (3, 5, 7) or None
        layer_positions: List of layer positions where attention is applied (1-indexed)
    
    Returns:
        Model name string, e.g., 'OM-Att-CH-2-L1', 'LM-Att-CBAM-4-5-L1-2-3'
    """
    # Layer positions string
    layers_str = "-".join([f"L{i}" for i in sorted(layer_positions)])
    
    if attention_type == "CH_ATT":
        if reduction_ratio is None:
            raise ValueError("reduction_ratio required for CH_ATT")
        return f"{variant}-Att-CH-{reduction_ratio}-{layers_str}"
    
    elif attention_type == "SP_ATT":
        if kernel_size is None:
            raise ValueError("kernel_size required for SP_ATT")
        return f"{variant}-Att-SP-{kernel_size}-{layers_str}"
    
    elif attention_type == "CBAM":
        if reduction_ratio is None or kernel_size is None:
            raise ValueError("reduction_ratio and kernel_size required for CBAM")
        return f"{variant}-Att-CBAM-{reduction_ratio}-{kernel_size}-{layers_str}"
    
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


def generate_kd_model_name(
    variant: str,
    kd_type: str,
    temperature: Optional[int] = None,
    alpha: Optional[float] = None,
    attention_config: Optional[str] = None,
) -> str:
    """
    Generate model name for Knowledge Distillation models.
    
    Args:
        variant: 'LM' (student variant)
        kd_type: 'RB-KD', 'RB-KD-Att', or 'RAB-KD-Att'
        temperature: Temperature value (1, 2, 4, 6, 8, 10, 15) or None
        alpha: Alpha value (0.1 to 0.9) or None
        attention_config: Attention config string (for RB-KD-Att and RAB-KD-Att) or None
    
    Returns:
        Model name string, e.g., 'LM-RB-KD-T10-A0.7', 'LM-RB-KD-Att-T4-A0.5-CH-2-L1'
    """
    parts = [variant, kd_type]
    
    if temperature is not None:
        parts.append(f"T{temperature}")
    
    if alpha is not None:
        parts.append(f"A{alpha:.1f}".replace(".", ""))  # A0.7 -> A07, A0.1 -> A01
    
    if attention_config:
        parts.append(attention_config)
    
    return "-".join(parts)

