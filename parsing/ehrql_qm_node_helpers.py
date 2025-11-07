import datetime
from enum import Enum
from functools import lru_cache
from typing import Any

from ehrql.query_model import nodes as qm


# Cache commonly accessed query model node classes.
_QM_NODE_BASE = getattr(qm, "Node", None)
_FILTER_NODE = getattr(qm, "Filter", None)
_SORT_NODE = getattr(qm, "Sort", None)
_AND_NODE = getattr(qm, "And", None)


def _is_qm_node(value: Any) -> bool:
    if _QM_NODE_BASE is not None:
        return isinstance(value, _QM_NODE_BASE)
    return hasattr(value, "__dataclass_fields__")


def _matches_filter(node: Any) -> bool:
    if _FILTER_NODE is not None and isinstance(node, _FILTER_NODE):
        return True
    return getattr(node.__class__, "__name__", "") == "Filter"


def _matches_sort(node: Any) -> bool:
    if _SORT_NODE is not None and isinstance(node, _SORT_NODE):
        return True
    return getattr(node.__class__, "__name__", "") == "Sort"


def _matches_and(node: Any) -> bool:
    if _AND_NODE is not None and isinstance(node, _AND_NODE):
        return True
    return getattr(node.__class__, "__name__", "") == "And"


def normalize_qm_node(qm_node: qm.Node) -> qm.Node:
    """Normalize a QM node by reordering Filter and Sort operations.

    Ensures Filters always come before Sorts, as they are semantically equivalent
    but we want a canonical ordering for comparison purposes.

    Recursively normalizes all nested nodes and applies transformations until
    a fixed point is reached.

    Transformations applied:
    1. Filter with And condition:
       Filter(source=X, condition=And(lhs=L, rhs=R))
       becomes:
       Filter(source=Filter(source=X, condition=L), condition=R)

    2. Filter with Sort source:
       Filter(source=Sort(source=X, sort_by=Y), condition=Z)
       becomes:
       Sort(source=Filter(source=X, condition=Z), sort_by=Y)

    Example: Filter(Filter(Sort(X))) becomes Sort(Filter(Filter(X)))
    """

    def normalize_once(node: qm.Node) -> tuple[qm.Node, bool]:
        """Apply one pass, return (node, changed)."""
        changed = False

        if hasattr(node, "__dataclass_fields__"):
            normalized_fields = {}
            for field_name in node.__dataclass_fields__:
                field_value = getattr(node, field_name)
                if _is_qm_node(field_value):
                    normalized_value, child_changed = normalize_once(field_value)
                    normalized_fields[field_name] = normalized_value
                    changed = changed or child_changed
                else:
                    normalized_fields[field_name] = field_value

            if changed:
                node = node.__class__(**normalized_fields)

        if _matches_filter(node):
            filter_source = node.source
            filter_condition = node.condition

            if _matches_and(filter_condition):
                conditions: list[qm.Node] = []
                stack = [filter_condition]
                while stack:
                    current = stack.pop()
                    if _matches_and(current):
                        stack.append(current.rhs)
                        stack.append(current.lhs)
                    else:
                        conditions.append(current)

                if len(conditions) > 1:
                    filter_cls = node.__class__
                    new_source = filter_source
                    for cond in conditions[:-1]:
                        new_source = filter_cls(source=new_source, condition=cond)
                    return filter_cls(source=new_source, condition=conditions[-1]), True

            if _matches_sort(filter_source):
                sort_node = filter_source
                filter_cls = node.__class__
                sort_cls = sort_node.__class__
                new_filter = filter_cls(
                    source=sort_node.source, condition=filter_condition
                )
                return sort_cls(source=new_filter, sort_by=sort_node.sort_by), True

        return node, changed

    # Keep applying normalization until we reach a fixed point
    # (i.e., no more changes occur)
    max_iterations = 100  # Safety limit to prevent infinite loops
    # prev_node = None
    current_node = qm_node

    for _ in range(max_iterations):
        current_node, is_changed = normalize_once(current_node)
        # Check if we've reached a fixed point by comparing string representations
        # (comparing objects directly won't work as they're new instances)
        # if prev_node is not None and str(current_node) == str(prev_node):
        #     break
        if not is_changed:
            break
        # prev_node = current_node

    return current_node


@lru_cache(maxsize=4096)
def _stringify_frozenset(value) -> str:
    # canonical key is a tuple of sorted reprs
    key = tuple(sorted(map(repr, value)))
    return f"frozenset({{{', '.join(key)}}})"


def _stringify_value(value):
    """Convert a value to a deterministic string representation, handling frozensets specially."""
    if isinstance(value, frozenset):
        return _stringify_frozenset(value)
    else:
        return str(value)


def compact_qm_node(qm_node: qm.Node, _normalized: bool = False) -> str:
    # Navigate all dataclass fields of each node recursively
    # When encountering a SelectTable, replace it entirely with the string from the table name
    try:
        # Handle sets of nodes (e.g., Domain sets)
        if isinstance(qm_node, set):
            # Sort by string representation for consistent output
            sorted_nodes = sorted(qm_node, key=lambda n: str(n))
            return (
                "{"
                + ", ".join(
                    compact_qm_node(node, _normalized=True) for node in sorted_nodes
                )
                + "}"
            )

        # First, normalize the node structure (e.g., reorder Filter/Sort)
        # Only do this at the top level, not recursively
        if not _normalized:
            try:
                qm_node = normalize_qm_node(qm_node)
            except Exception:
                # If normalization fails (e.g., ehrql raises "Attempt to combine unrelated domains"),
                # skip normalization and use the original node
                pass

        if isinstance(qm_node, qm.SelectTable):
            return f"Table({qm_node.name})"
        elif isinstance(qm_node, qm.SelectPatientTable):
            return f"Table({qm_node.name})"
        elif isinstance(qm_node, qm.Node):
            # Canonicalize commutative binary operations (Or, And) by flattening chains and sorting
            if (
                qm_node.__class__.__name__ in ("Or", "And")
                and hasattr(qm_node, "lhs")
                and hasattr(qm_node, "rhs")
            ):
                op_name = qm_node.__class__.__name__

                # Flatten chains of the same operation (Or-of-Or or And-of-And)
                def flatten_op(node, op_type):
                    """Recursively flatten chains of the same commutative operation."""
                    if (
                        isinstance(node, qm.Node)
                        and node.__class__.__name__ == op_type
                        and hasattr(node, "lhs")
                        and hasattr(node, "rhs")
                    ):
                        # Recursively flatten both sides
                        return flatten_op(node.lhs, op_type) + flatten_op(
                            node.rhs, op_type
                        )
                    else:
                        return [node]

                # Flatten the chain into a list of operands
                operands = flatten_op(qm_node, op_name)

                # Compact each operand and sort them
                operand_strs = [
                    compact_qm_node(op, _normalized=True) for op in operands
                ]
                operand_strs.sort()

                # Rebuild as a right-associated binary tree
                if len(operand_strs) == 1:
                    return operand_strs[0]

                result = operand_strs[0]
                for i in range(1, len(operand_strs)):
                    result = f"{op_name}(lhs={result}, rhs={operand_strs[i]})"
                return result

            fields = {}
            for field_name in list(qm_node.__dataclass_fields__):
                field_value = getattr(qm_node, field_name)
                if isinstance(field_value, qm.Node):
                    fields[field_name] = compact_qm_node(field_value, _normalized=True)
                elif field_name == "cases" and isinstance(field_value, dict):
                    # Sort cases by canonicalized key string for determinism
                    # Compute key strings once and cache them to avoid redundant computation
                    key_strs = []
                    for k, v in field_value.items():
                        if isinstance(k, qm.Node):
                            k_str = compact_qm_node(k, _normalized=True)
                        else:
                            k_str = str(k)
                        if isinstance(v, qm.Node):
                            v_str = compact_qm_node(v, _normalized=True)
                        else:
                            v_str = _stringify_value(v)
                        key_strs.append((k_str, v_str))

                    # Sort by the pre-computed key strings
                    key_strs.sort(key=lambda x: x[0])

                    # Build the output string
                    parts = [f"if:{k_str}->then:{v}" for k_str, v in key_strs]
                    fields[field_name] = ", ".join(parts)
                elif isinstance(field_value, list):
                    # If list of Nodes, preserve order and compact each.
                    if all(isinstance(item, qm.Node) for item in field_value):
                        parts = [
                            compact_qm_node(item, _normalized=True)
                            for item in field_value
                        ]
                    else:
                        # If list of scalars (e.g., codes), sort deterministically by string value
                        if all(
                            not isinstance(item, qm.Node)
                            and isinstance(item, (str, int, float))
                            for item in field_value
                        ):
                            sorted_vals = sorted(field_value, key=lambda x: str(x))
                            parts = [_stringify_value(x) for x in sorted_vals]
                        else:
                            parts = [_stringify_value(x) for x in field_value]
                    fields[field_name] = ", ".join(parts)
                elif isinstance(field_value, tuple):
                    # Apply the same determinism to tuples of scalars
                    if all(isinstance(item, qm.Node) for item in field_value):
                        parts = [
                            compact_qm_node(item, _normalized=True)
                            for item in field_value
                        ]
                    else:
                        if all(
                            not isinstance(item, qm.Node)
                            and isinstance(item, (str, int, float))
                            for item in field_value
                        ):
                            sorted_vals = sorted(field_value, key=lambda x: str(x))
                            parts = [_stringify_value(x) for x in sorted_vals]
                        else:
                            parts = [_stringify_value(x) for x in field_value]
                    fields[field_name] = ", ".join(parts)
                elif isinstance(field_value, set):
                    # Handle sets (e.g., Domain sets)
                    sorted_items = sorted(field_value, key=lambda x: str(x))
                    fields[field_name] = (
                        "{"
                        + ", ".join(
                            [
                                compact_qm_node(item, _normalized=True)
                                if isinstance(item, qm.Node)
                                else _stringify_value(item)
                                for item in sorted_items
                            ]
                        )
                        + "}"
                    )
                elif isinstance(field_value, datetime.date):
                    fields[field_name] = "{{DATE}}"
                elif isinstance(field_value, str):
                    fields[field_name] = field_value
                elif isinstance(field_value, frozenset):
                    # Sort by repr string to avoid relying on object __lt__ implementations
                    # sorted_strs = sorted(repr(x) for x in field_value)
                    # fields[field_name] = f"frozenset({{{', '.join(sorted_strs)}}})"
                    fields[field_name] = _stringify_frozenset(field_value)
                elif isinstance(field_value, Enum):
                    fields[field_name] = field_value.name
                elif isinstance(field_value, int):
                    fields[field_name] = field_value
                elif isinstance(field_value, float):
                    fields[field_name] = field_value
                else:
                    # For any other type, use our stringify helper to ensure determinism
                    fields[field_name] = _stringify_value(field_value)
            field_strs = [f"{k}={v}" for k, v in fields.items()]
            result = f"{qm_node.__class__.__name__}({', '.join(field_strs)})"
            return result
    except Exception as e:
        print(f"Error compacting QM node: {e}")
        return str(qm_node)
