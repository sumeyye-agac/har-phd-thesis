"""
Generic command-line argument parser for grid search scripts.

This module provides a unified parser for all grid search scripts,
eliminating code duplication and centralizing parsing logic.
"""

import sys
from typing import Dict, List, Any, Optional


def parse_comma_separated_list(
    value: str,
    valid_values: Optional[List[Any]] = None,
    param_name: str = ""
) -> List[Any]:
    """
    Parse comma-separated values like '1,2,3' or '0.1,0.2'.
    
    Args:
        value: Comma-separated string
        valid_values: Optional list of valid values for validation
        param_name: Parameter name for error messages
    
    Returns:
        List of parsed values (int or float)
    
    Raises:
        SystemExit: On parsing or validation errors
    """
    try:
        # Split by comma and strip whitespace
        value_parts = [v.strip() for v in value.split(',') if v.strip()]
        if not value_parts:
            raise ValueError("Empty list")
        
        # Convert to appropriate type
        parsed_list = []
        for v in value_parts:
            try:
                # Try int first
                parsed_val = int(v)
            except ValueError:
                try:
                    # Try float
                    parsed_val = float(v)
                except ValueError:
                    print(f"❌ Error: Invalid value in '{param_name}': '{v}' (not a number)")
                    print()
                    if valid_values:
                        print(f"   Valid values: {', '.join(map(str, valid_values))}")
                        print(f"   Example: {param_name}={valid_values[0]},{valid_values[1]}")
                    sys.exit(1)
            parsed_list.append(parsed_val)
        
        # Validate against valid_values if provided
        if valid_values is not None:
            invalid = [v for v in parsed_list if v not in valid_values]
            if invalid:
                print(f"❌ Error: Invalid value in '{param_name}': {invalid[0]}")
                print()
                print(f"   Valid values: {', '.join(map(str, valid_values))}")
                if len(valid_values) >= 2:
                    print(f"   Example: {param_name}={valid_values[0]},{valid_values[1]}")
                sys.exit(1)
        
        return parsed_list
        
    except Exception as e:
        print(f"❌ Error: Invalid format for '{param_name}': '{value}'")
        print()
        print("   Valid format: value1,value2,... (comma-separated values)")
        if valid_values:
            print(f"   Valid values: {', '.join(map(str, valid_values))}")
            if len(valid_values) >= 2:
                print(f"   Example: {param_name}={valid_values[0]},{valid_values[1]}")
        sys.exit(1)


def parse_boolean(value: str, param_name: str = "") -> bool:
    """
    Parse boolean values like 'true' or 'false'.
    
    Args:
        value: String value ('true' or 'false', case-insensitive)
        param_name: Parameter name for error messages
    
    Returns:
        Boolean value
    
    Raises:
        SystemExit: On invalid values
    """
    value_lower = value.lower()
    if value_lower not in ('true', 'false'):
        print(f"❌ Error: Invalid value for '{param_name}': '{value}'")
        print()
        print("   Valid values: true, false")
        print(f"   Example: {param_name}=true or {param_name}=false")
        sys.exit(1)
    return value_lower == 'true'


def parse_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse command-line arguments based on provided configuration.
    
    Args:
        config: Configuration dictionary with:
            - 'attention_params': List of boolean parameter names (optional)
            - 'list_params': List of list parameter names
            - 'valid_values': Dict mapping param_name -> list of valid values
            - 'help_text': Help message string
            - 'script_name': Script name for error messages
    
    Returns:
        Dictionary of parsed arguments
    
    Raises:
        SystemExit: On parsing errors or --help flag
    """
    # Extract config
    attention_params = config.get('attention_params', [])
    list_params = config.get('list_params', [])
    valid_values = config.get('valid_values', {})
    help_text = config.get('help_text', '')
    script_name = config.get('script_name', 'script.py')
    
    all_params = attention_params + list_params
    
    # Check for --help
    if '--help' in sys.argv or '-h' in sys.argv:
        print(help_text)
        sys.exit(0)
    
    parsed_config = {}
    
    for arg in sys.argv[1:]:
        if '=' not in arg:
            print(f"❌ Error: Invalid argument format: '{arg}'")
            print("   Expected format: parameter=value")
            print("   Example: ch_att=true")
            print()
            print(help_text)
            sys.exit(1)
        
        key, value = arg.split('=', 1)
        key_lower = key.lower()
        
        # Check if parameter is valid
        if key_lower not in all_params:
            print(f"❌ Error: Unknown parameter: '{key}'")
            print()
            print("   Valid parameters:")
            if attention_params:
                print(f"   - {', '.join(attention_params)} (boolean: true/false)")
            if list_params:
                print(f"   - {', '.join(list_params)} (list: value1,value2,...)")
            print()
            print(f"   Example: python {script_name} {' '.join(attention_params[:1] if attention_params else list_params[:1])}=...")
            sys.exit(1)
        
        # Parse boolean values
        if key_lower in attention_params:
            parsed_config[key_lower] = parse_boolean(value, key)
        
        # Parse list values
        elif key_lower in list_params:
            param_valid_values = valid_values.get(key_lower)
            parsed_config[key_lower] = parse_comma_separated_list(
                value,
                valid_values=param_valid_values,
                param_name=key
            )
    
    return parsed_config

