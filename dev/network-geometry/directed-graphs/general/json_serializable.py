import json
import jax.numpy as jnp
from jax.core import Tracer
def make_json_serializable(obj):
    """
    Convert an object to a JSON-serializable form, including handling JAX types.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        # These types are natively serializable
        return obj
    elif isinstance(obj, dict):
        # Recursively apply to all dictionary items
        # if k can be integer keep, as integer
        try:
            return {int(k): make_json_serializable(v) for k, v in obj.items()}
        except:
            return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively apply to all list elements
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuples to lists
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert sets to lists
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, jnp.ndarray):
        # Convert JAX arrays to lists
        return obj.tolist()
    elif isinstance(obj, Tracer):
        # Handle JAX tracers by converting them to regular NumPy arrays
        return make_json_serializable(jnp.asarray(obj))
    elif hasattr(obj, "__dict__"):
        # Handle custom objects by converting their __dict__ attribute
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
    else:
        # Fallback for other types (e.g., classes without __dict__)
        return str(obj)