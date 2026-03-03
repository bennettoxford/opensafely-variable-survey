"""Extract dataset variable definitions from ehrQL Python files using AST analysis.

This module provides a cleaner, more maintainable implementation of variable extraction
with clear separation of concerns:
- Module resolution (finding imported modules)
- Pattern extraction (finding dataset operations in AST)
- Dynamic name handling (f-strings, BinOp concatenation)
"""

from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass


@dataclass
class VariableLocation:
    """Location of a variable definition."""

    name: str
    line_number: int
    file_path: pathlib.Path | None = None  # None means same file as main


@dataclass
class DynamicVariablePattern:
    """Pattern for dynamically-named variables (f-strings, loops, etc)."""

    regex_pattern: str
    line_number: int


@dataclass
class CodelistCall:
    """Represents a call to codelist_from_csv()."""

    args: tuple[
        str | None, ...
    ]  # Positional arguments (first is usually the file path)
    kwargs: dict[
        str, str | None
    ]  # Keyword arguments like system, column, category_column


class ModuleResolver:
    """Resolves imported module names to file paths."""

    def __init__(self, file_path: pathlib.Path, repo_root: pathlib.Path):
        self.file_path = file_path
        self.repo_root = repo_root
        self.parent_dir = file_path.parent

    def find_module_file(self, module_name: str) -> list[pathlib.Path]:
        """Find candidate file paths for a module name.

        Args:
            module_name: Dotted module name like "helpers" or "analysis.utils"

        Returns:
            List of candidate paths (first existing one should be used)
        """
        if not module_name:
            return []

        module_parts = module_name.split(".")
        module_rel_path = pathlib.Path(*module_parts)

        # Build candidates walking up the directory tree
        # This handles cases where current file is already in a subdir
        candidate_files: list[pathlib.Path] = []
        for ancestor in [self.parent_dir, *self.parent_dir.parents]:
            base = ancestor / module_rel_path
            candidate_files.append(base.with_suffix(".py"))
            candidate_files.append(base / "__init__.py")
            if ancestor == ancestor.parent:
                break

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_candidates: list[pathlib.Path] = []
        for candidate in candidate_files:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        return unique_candidates

    def get_relative_path(self, file_path: pathlib.Path) -> str:
        """Get path relative to repo root, or str(path) if not in repo."""
        try:
            return str(file_path.relative_to(self.repo_root))
        except ValueError:
            return str(file_path)


class DynamicNameExtractor:
    """Extracts regex patterns from dynamic variable names."""

    @staticmethod
    def extract_from_fstring(fstring_node: ast.JoinedStr) -> str:
        """Convert f-string AST node to regex pattern.

        Example: f"age_{i}" -> "age_.*"
        """
        parts: list[str] = []
        for value in fstring_node.values:
            if isinstance(value, ast.Constant):
                parts.append(re.escape(str(value.value)))
            else:
                parts.append(".*")
        return "".join(parts)

    @staticmethod
    def extract_from_binop(binop_node: ast.BinOp) -> str | None:
        """Convert BinOp string concatenation to regex pattern.

        Example: "age_" + suffix -> "age_.*"
        Returns None if not a simple Add operation on strings.
        """
        if not isinstance(binop_node.op, ast.Add):
            return None

        parts: list[str] = []

        def extract_parts(node: ast.AST) -> None:
            if isinstance(node, ast.Constant):
                parts.append(re.escape(str(node.value)))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                extract_parts(node.left)
                extract_parts(node.right)
            else:
                # Unknown/dynamic part
                parts.append(".*")

        extract_parts(binop_node)
        return "".join(parts)


class CodelistCallFinder:
    """Finds codelist_from_csv() calls in AST nodes."""

    @staticmethod
    def extract_codelist_calls(tree: ast.AST) -> dict[str, list[CodelistCall]]:
        """Extract all codelist_from_csv calls, organized by variable name.

        Args:
            tree: AST tree to search

        Returns:
            Dict mapping variable_name -> list of CodelistCall objects
        """
        codelist_calls: dict[str, list[CodelistCall]] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue

            # Get the variable name from the assignment target
            var_name: str | None = None
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id

            if not var_name:
                continue

            # Check if the RHS is a call to codelist_from_csv
            if not isinstance(node.value, ast.Call):
                continue

            func = node.value.func
            is_codelist_call = False

            # Check for direct call: codelist_from_csv(...)
            if isinstance(func, ast.Name) and func.id == "codelist_from_csv":
                is_codelist_call = True
            # Check for module.codelist_from_csv(...)
            elif isinstance(func, ast.Attribute) and func.attr == "codelist_from_csv":
                is_codelist_call = True

            if not is_codelist_call:
                continue

            # Extract positional arguments
            args_list: list[str | None] = []
            for arg in node.value.args:
                if isinstance(arg, ast.Constant):
                    args_list.append(str(arg.value))
                else:
                    args_list.append(None)  # Non-constant argument

            # Extract keyword arguments
            kwargs_dict: dict[str, str | None] = {}
            for keyword in node.value.keywords:
                if keyword.arg:
                    if isinstance(keyword.value, ast.Constant):
                        kwargs_dict[keyword.arg] = str(keyword.value.value)
                    else:
                        kwargs_dict[keyword.arg] = None  # Non-constant value

            codelist_call = CodelistCall(
                args=tuple(args_list),
                kwargs=kwargs_dict,
            )

            if var_name not in codelist_calls:
                codelist_calls[var_name] = []
            codelist_calls[var_name].append(codelist_call)

        return codelist_calls


class DatasetOperationFinder:
    """Finds dataset operations in AST nodes."""

    def __init__(self, name_extractor: DynamicNameExtractor):
        self.name_extractor = name_extractor

    def find_variable_assignments(
        self,
        node: ast.AST,
        dataset_name: str = "dataset",
    ) -> list[tuple[str, int]]:
        """Find dataset.variable = expression patterns.

        Returns:
            List of (variable_name, line_number) tuples
        """
        results: list[tuple[str, int]] = []

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id == dataset_name
                    ):
                        results.append((target.attr, node.lineno))

        return results

    def find_add_column_calls(
        self,
        node: ast.AST,
        dataset_name: str = "dataset",
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Find dataset.add_column(...) calls.

        Returns:
            Tuple of (static_vars, dynamic_patterns)
            where each is a list of (name/pattern, line_number)
        """
        static_vars: list[tuple[str, int]] = []
        dynamic_patterns: list[tuple[str, int]] = []

        if not isinstance(node, ast.Call):
            return static_vars, dynamic_patterns

        if not isinstance(node.func, ast.Attribute):
            return static_vars, dynamic_patterns

        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == dataset_name
            and node.func.attr == "add_column"
        ):
            return static_vars, dynamic_patterns

        if not node.args or len(node.args) < 1:
            return static_vars, dynamic_patterns

        first_arg = node.args[0]

        # Handle static string literal
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            static_vars.append((first_arg.value, node.lineno))

        # Handle Name node (parameter that needs to be resolved from call site)
        elif isinstance(first_arg, ast.Name):
            # Return the parameter name so it can be resolved later
            static_vars.append((first_arg.id, node.lineno))

        # Handle f-string
        elif isinstance(first_arg, ast.JoinedStr):
            pattern = self.name_extractor.extract_from_fstring(first_arg)
            dynamic_patterns.append((pattern, node.lineno))

        # Handle BinOp concatenation
        elif isinstance(first_arg, ast.BinOp):
            pattern = self.name_extractor.extract_from_binop(first_arg)
            if pattern:
                dynamic_patterns.append((pattern, node.lineno))

        return static_vars, dynamic_patterns

    def find_setattr_calls(
        self,
        node: ast.AST,
        dataset_name: str = "dataset",
        func_scope: ast.FunctionDef | None = None,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Find setattr(dataset, name, value) calls.

        Args:
            node: AST node to search
            dataset_name: Name of the dataset variable
            func_scope: Optional function definition to resolve variable assignments

        Returns:
            Tuple of (static_vars, dynamic_patterns)
        """
        static_vars: list[tuple[str, int]] = []
        dynamic_patterns: list[tuple[str, int]] = []

        if not isinstance(node, ast.Call):
            return static_vars, dynamic_patterns

        if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
            return static_vars, dynamic_patterns

        if len(node.args) < 2:
            return static_vars, dynamic_patterns

        obj_arg = node.args[0]
        name_arg = node.args[1]

        if not (isinstance(obj_arg, ast.Name) and obj_arg.id == dataset_name):
            return static_vars, dynamic_patterns

        # Handle static string literal
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            static_vars.append((name_arg.value, node.lineno))

        # Handle f-string
        elif isinstance(name_arg, ast.JoinedStr):
            pattern = self.name_extractor.extract_from_fstring(name_arg)
            dynamic_patterns.append((pattern, node.lineno))

        # Handle BinOp concatenation
        elif isinstance(name_arg, ast.BinOp):
            pattern = self.name_extractor.extract_from_binop(name_arg)
            if pattern:
                dynamic_patterns.append((pattern, node.lineno))

        # Handle Name node (variable reference) - try to resolve it
        elif isinstance(name_arg, ast.Name) and func_scope is not None:
            # Try to find if this variable was assigned from a .format() call
            pattern = self._resolve_template_variable(name_arg.id, func_scope)
            if pattern:
                dynamic_patterns.append((pattern, node.lineno))

        return static_vars, dynamic_patterns

    def _resolve_template_variable(
        self, var_name: str, func_scope: ast.FunctionDef
    ) -> str | None:
        """Resolve a variable that might be assigned from a template.format() call.

        Args:
            var_name: Name of the variable to resolve
            func_scope: Function definition containing the variable

        Returns:
            Regex pattern if the variable is assigned from a .format() call, else None
        """
        # Look for assignments like: variable_name = template.format(...)
        for node in ast.walk(func_scope):
            if not isinstance(node, ast.Assign):
                continue

            # Check if this assigns to our variable
            assigns_to_var = False
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    assigns_to_var = True
                    break

            if not assigns_to_var:
                continue

            # Check if the value is a .format() call
            if not isinstance(node.value, ast.Call):
                continue

            if not isinstance(node.value.func, ast.Attribute):
                continue

            if node.value.func.attr != "format":
                continue

            # The object being .format() called on should be a Name (parameter)
            if not isinstance(node.value.func.value, ast.Name):
                continue

            # Extract the template parameter name
            template_param = node.value.func.value.id

            # Check if this parameter is in the function signature
            param_names = [arg.arg for arg in func_scope.args.args]
            if template_param not in param_names:
                continue

            # We found a pattern like: var = template_param.format(...)
            # The template parameter holds a string with placeholders like "admission{n}_date_sus"
            # We need to convert this to a regex pattern that matches the expected variables
            # Since we don't know the actual template value at AST parse time, we mark it
            # as a generic pattern. The actual template will be resolved from the call site
            # in the calling code.
            return "__TEMPLATE_FORMAT__"  # Marker for template.format() pattern

        return None

    def find_subscript_assignments(
        self,
        node: ast.AST,
        dataset_name: str = "dataset",
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Find dataset[name] = value patterns.

        Returns:
            Tuple of (static_vars, dynamic_patterns)
        """
        static_vars: list[tuple[str, int]] = []
        dynamic_patterns: list[tuple[str, int]] = []

        if not isinstance(node, ast.Assign):
            return static_vars, dynamic_patterns

        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue

            if not (
                isinstance(target.value, ast.Name) and target.value.id == dataset_name
            ):
                continue

            slice_node = getattr(target, "slice", None)
            if not slice_node:
                continue

            # Handle static string
            if isinstance(slice_node, ast.Constant) and isinstance(
                slice_node.value, str
            ):
                static_vars.append((slice_node.value, node.lineno))

            # Handle f-string
            elif isinstance(slice_node, ast.JoinedStr):
                pattern = self.name_extractor.extract_from_fstring(slice_node)
                dynamic_patterns.append((pattern, node.lineno))

        return static_vars, dynamic_patterns

    def find_setattr_with_param_index(
        self,
        func_def: ast.FunctionDef,
        dataset_param_name: str,
    ) -> int | None:
        """Find if function uses setattr(dataset, param_name, ...) pattern.

        Returns the parameter index that holds the variable name, or None.
        This is for patterns like: has_prior_comorbidity(extract_name, ..., dataset)
        where the function does setattr(dataset, extract_name, value).
        """
        for node in ast.walk(func_def):
            if not isinstance(node, ast.Call):
                continue

            # Check for setattr call
            if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
                continue

            if len(node.args) < 2:
                continue

            # First arg should be dataset parameter
            obj_arg = node.args[0]
            if not (isinstance(obj_arg, ast.Name) and obj_arg.id == dataset_param_name):
                continue

            # Second arg is the variable name - check if it's a parameter
            name_arg = node.args[1]
            if isinstance(name_arg, ast.Name):
                # Find which parameter this is
                for idx, param in enumerate(func_def.args.args):
                    if param.arg == name_arg.id:
                        return idx

        return None


class FunctionAnalyzer:
    """Analyzes function definitions to extract dataset operations."""

    def __init__(self, operation_finder: DatasetOperationFinder):
        self.operation_finder = operation_finder

    def extract_from_function(
        self,
        func_def: ast.FunctionDef,
        dataset_param_name: str,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Extract all dataset operations from a function definition.

        Args:
            func_def: Function AST node
            dataset_param_name: Name of the parameter that represents the dataset

        Returns:
            Tuple of (static_vars, dynamic_patterns) where each is list of (name, line)
        """
        static_vars: list[tuple[str, int]] = []
        dynamic_patterns: list[tuple[str, int]] = []

        for node in ast.walk(func_def):
            # Check for attribute assignments
            for var_name, line in self.operation_finder.find_variable_assignments(
                node, dataset_param_name
            ):
                static_vars.append((var_name, line))

            # Check for add_column calls
            static, dynamic = self.operation_finder.find_add_column_calls(
                node, dataset_param_name
            )
            static_vars.extend(static)
            dynamic_patterns.extend(dynamic)

            # Check for setattr calls - pass func_scope to resolve template variables
            static, dynamic = self.operation_finder.find_setattr_calls(
                node, dataset_param_name, func_scope=func_def
            )
            static_vars.extend(static)
            dynamic_patterns.extend(dynamic)

            # Check for subscript assignments
            static, dynamic = self.operation_finder.find_subscript_assignments(
                node, dataset_param_name
            )
            static_vars.extend(static)
            dynamic_patterns.extend(dynamic)

        return static_vars, dynamic_patterns

    def find_dataset_param_index(
        self,
        func_def: ast.FunctionDef,
        call_node: ast.Call,
    ) -> int | None:
        """Determine which parameter of func_def corresponds to the dataset.

        Checks both positional and keyword arguments in the call.

        Returns:
            Parameter index, or None if dataset not found
        """
        # Check positional arguments for "dataset"
        for idx, arg in enumerate(call_node.args):
            if isinstance(arg, ast.Name) and arg.id == "dataset":
                return idx

        # Check keyword arguments
        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Name) and kw.value.id == "dataset":
                # Find this keyword in the function definition
                for idx, param in enumerate(func_def.args.args):
                    if param.arg == kw.arg:
                        return idx
                # Also handle if keyword matches dataset param directly
                if kw.arg == "dataset":
                    for idx, param in enumerate(func_def.args.args):
                        if param.arg == "dataset":
                            return idx

        return None


class ImportCollector:
    """Collects import information from an AST."""

    def __init__(self):
        self.function_defs: dict[str, ast.FunctionDef] = {}
        self.class_defs: dict[str, ast.ClassDef] = {}
        self.imported_modules: dict[str, tuple[str, str | None]] = {}
        self.star_imports: list[str] = []

    def collect(self, tree: ast.AST) -> None:
        """Walk AST and collect all imports, function definitions, and class definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.function_defs[node.name] = node

            elif isinstance(node, ast.ClassDef):
                self.class_defs[node.name] = node

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        self.star_imports.append(module_name)
                        if module_name:
                            base_name = module_name.split(".")[-1]
                            if base_name:
                                self.imported_modules.setdefault(
                                    base_name, (module_name, None)
                                )
                    else:
                        imported_name = alias.asname if alias.asname else alias.name
                        self.imported_modules[imported_name] = (module_name, alias.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name
                    self.imported_modules[imported_name] = (alias.name, None)

    def resolve_star_imports(self, module_resolver: ModuleResolver) -> None:
        """Resolve star imports by loading the modules and finding functions and variables."""
        for star_module in self.star_imports:
            for module_file in module_resolver.find_module_file(star_module):
                if not module_file.exists():
                    continue

                try:
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=str(module_file))

                    # Add all functions and module-level variables from this module
                    for node in ast.walk(module_tree):
                        if isinstance(node, ast.FunctionDef):
                            # Map function name to its module
                            self.imported_modules[node.name] = (star_module, node.name)
                        # Also track module-level variable assignments (like codelist definitions)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    # Map variable name to its module
                                    self.imported_modules[target.id] = (
                                        star_module,
                                        target.id,
                                    )
                    break  # Only process first found file
                except Exception:
                    continue


class ASTIndex:
    """Pre-computed index of AST nodes for fast lookup."""

    def __init__(self, tree: ast.AST):
        # Maps for fast lookup
        self.var_assignments: dict[
            str, list[ast.Assign]
        ] = {}  # var_name -> [assign_nodes]
        self.dataset_assignments: dict[
            str, ast.Assign
        ] = {}  # attr_name -> assign_node (dataset.attr = ...)
        self.class_defs: dict[str, ast.ClassDef] = {}  # class_name -> class_def

        # Build indexes
        self._build_indexes(tree)

    def _build_indexes(self, tree: ast.AST) -> None:
        """Walk the tree once and build all indexes."""
        # Only walk top-level and function-level nodes
        nodes_to_check: list[ast.AST] = []
        if isinstance(tree, ast.Module):
            nodes_to_check.extend(tree.body)
            # Also check inside function definitions for local assignments
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    nodes_to_check.extend(node.body)
                elif isinstance(node, ast.ClassDef):
                    # Index class definitions
                    self.class_defs[node.name] = node
        else:
            nodes_to_check = list(ast.walk(tree))

        for node in nodes_to_check:
            # Index variable assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # Regular variable assignment: var = expr
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name not in self.var_assignments:
                            self.var_assignments[var_name] = []
                        self.var_assignments[var_name].append(node)

                    # Dataset attribute assignment: dataset.attr = expr
                    elif isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "dataset"
                        ):
                            self.dataset_assignments[target.attr] = node


class CodelistTracer:
    """Traces codelist_from_csv calls through variable and function calls."""

    INLINE_CODELIST_SENTINEL = "<inline>"

    def __init__(self, module_resolver: ModuleResolver):
        self.module_resolver = module_resolver
        self._visited_vars: set[tuple[str, str]] = set()  # (file_path, var_name)
        self._codelist_cache: dict[tuple[str, str], list[CodelistCall]] = {}
        self._module_tree_cache: dict[
            str, tuple[ast.AST, ImportCollector, ASTIndex]
        ] = {}  # file_path -> (tree, imports, index)

    def trace_expression_for_codelists(
        self,
        expr: ast.AST,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
    ) -> list[CodelistCall]:
        """Trace an expression comprehensively to find all codelist_from_csv calls.

        This is the main entry point for tracing a dataset variable's expression.

        Args:
            expr: The expression AST node (RHS of variable assignment)
            tree: The full AST tree of the file
            import_collector: Import information
            file_path: Path to the file being analyzed

        Returns:
            List of all CodelistCall objects found in the expression's call tree
        """
        self._visited_vars.clear()
        # Create index for this tree if not already cached
        ast_index = ASTIndex(tree)
        return self._trace_expression(
            expr, tree, import_collector, file_path, ast_index
        )

    def _trace_expression(
        self,
        expr: ast.AST,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int = 0,
    ) -> list[CodelistCall]:
        """Recursively trace an expression to find all codelist_from_csv calls.

        Args:
            expr: Expression to trace
            tree: AST tree of current file
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index for fast lookups
            depth: Recursion depth (for debugging/limits)

        Returns:
            List of CodelistCall objects
        """
        if depth > 50:  # Prevent infinite recursion
            return []

        codelist_calls: list[CodelistCall] = []

        # Walk through all nodes in the expression
        for node in ast.walk(expr):
            inline_call = self._extract_inline_literal(node, file_path)
            if inline_call is not None:
                codelist_calls.append(inline_call)

            # Direct codelist_from_csv call
            if isinstance(node, ast.Call):
                calls = self._check_for_codelist_call(node)
                codelist_calls.extend(calls)

            # Name reference - could be a variable holding a codelist
            if isinstance(node, ast.Name):
                calls = self._trace_name(
                    node.id, tree, import_collector, file_path, ast_index, depth
                )
                codelist_calls.extend(calls)

            # Attribute access - could be accessing codelist from class/module
            if isinstance(node, ast.Attribute):
                calls = self._trace_attribute(
                    node, tree, import_collector, file_path, ast_index, depth
                )
                codelist_calls.extend(calls)

        return codelist_calls

    def _extract_inline_literal(
        self, node: ast.AST, file_path: pathlib.Path
    ) -> CodelistCall | None:
        """Detect inline lists/tuples of literal codes and treat them as codelists."""

        literal_nodes = (ast.List, ast.Tuple, ast.Set)
        if not isinstance(node, literal_nodes):
            return None

        if not self._looks_like_codelist_file(file_path):
            return None

        values: list[str] = []
        for element in getattr(node, "elts", []):
            literal = self._literal_value_to_str(element)
            if literal is None:
                return None
            values.append(literal)

        if not values:
            return None

        rel_path = self.module_resolver.get_relative_path(file_path)
        line = getattr(node, "lineno", None)
        source = f"{rel_path}:{line}" if line is not None else rel_path
        kwargs = {
            "length": str(len(values)),
            "source": source,
            "values": "|".join(values),
        }
        return CodelistCall(
            args=(self.INLINE_CODELIST_SENTINEL,),
            kwargs=kwargs,
        )

    @staticmethod
    def _literal_value_to_str(node: ast.AST) -> str | None:
        """Convert a Constant literal to string if it looks like a code."""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float)):
                return str(node.value)
        return None

    @staticmethod
    def _looks_like_codelist_file(file_path: pathlib.Path) -> bool:
        """Heuristic to limit inline detection to codelist-focused modules."""

        return "codelist" in str(file_path).lower()

    def _check_for_codelist_call(self, call_node: ast.Call) -> list[CodelistCall]:
        """Check if a Call node is a codelist_from_csv call and extract it.

        Args:
            call_node: AST Call node to check

        Returns:
            List with single CodelistCall if this is a codelist_from_csv call, else empty list
        """
        func = call_node.func
        is_codelist_call = False

        # Check for direct call: codelist_from_csv(...)
        if isinstance(func, ast.Name) and func.id == "codelist_from_csv":
            is_codelist_call = True
        # Check for module.codelist_from_csv(...)
        elif isinstance(func, ast.Attribute) and func.attr == "codelist_from_csv":
            is_codelist_call = True

        if not is_codelist_call:
            return []

        # Extract arguments
        args_list: list[str | None] = []
        for arg in call_node.args:
            if isinstance(arg, ast.Constant):
                args_list.append(str(arg.value))
            else:
                args_list.append(None)

        # Extract keyword arguments
        kwargs_dict: dict[str, str | None] = {}
        for keyword in call_node.keywords:
            if keyword.arg:
                if isinstance(keyword.value, ast.Constant):
                    kwargs_dict[keyword.arg] = str(keyword.value.value)
                else:
                    kwargs_dict[keyword.arg] = None

        return [CodelistCall(args=tuple(args_list), kwargs=kwargs_dict)]

    def _check_for_codelist_call_in_enum(
        self, call_node: ast.Call, enum_member_values: tuple[str, ...]
    ) -> list[CodelistCall]:
        """Check if a Call node is a codelist_from_csv call in an Enum and extract it.

        This handles the pattern where codelists are defined in Enum __init__ with f-strings:
            f"codelists/{self.codelist_name}.csv"

        We resolve the f-string using the enum member's tuple value.

        Args:
            call_node: AST Call node to check
            enum_member_values: The tuple values from the enum member definition

        Returns:
            List with single CodelistCall if this is a codelist_from_csv call, else empty list
        """
        func = call_node.func
        is_codelist_call = False

        # Check for direct call: codelist_from_csv(...)
        if isinstance(func, ast.Name) and func.id == "codelist_from_csv":
            is_codelist_call = True
        # Check for module.codelist_from_csv(...)
        elif isinstance(func, ast.Attribute) and func.attr == "codelist_from_csv":
            is_codelist_call = True

        if not is_codelist_call:
            return []

        # Extract arguments, resolving f-strings
        args_list: list[str | None] = []
        for arg in call_node.args:
            if isinstance(arg, ast.Constant):
                args_list.append(str(arg.value))
            elif isinstance(arg, ast.JoinedStr):
                # This is an f-string - try to resolve it
                resolved = self._resolve_fstring_in_enum(arg, enum_member_values)
                args_list.append(resolved)
            else:
                args_list.append(None)

        # Extract keyword arguments, resolving f-strings
        kwargs_dict: dict[str, str | None] = {}
        for keyword in call_node.keywords:
            if keyword.arg:
                if isinstance(keyword.value, ast.Constant):
                    kwargs_dict[keyword.arg] = str(keyword.value.value)
                elif isinstance(keyword.value, ast.JoinedStr):
                    resolved = self._resolve_fstring_in_enum(
                        keyword.value, enum_member_values
                    )
                    kwargs_dict[keyword.arg] = resolved
                # Check if it's self._column or similar attribute access
                elif isinstance(keyword.value, ast.Attribute):
                    if (
                        isinstance(keyword.value.value, ast.Name)
                        and keyword.value.value.id == "self"
                    ):
                        # Map self._column to the second enum value, etc.
                        attr = keyword.value.attr
                        if attr == "_column" and len(enum_member_values) > 1:
                            kwargs_dict[keyword.arg] = enum_member_values[1]
                        else:
                            kwargs_dict[keyword.arg] = None
                    else:
                        kwargs_dict[keyword.arg] = None
                else:
                    kwargs_dict[keyword.arg] = None

        return [CodelistCall(args=tuple(args_list), kwargs=kwargs_dict)]

    def _resolve_fstring_in_enum(
        self, fstring: ast.JoinedStr, enum_member_values: tuple[str, ...]
    ) -> str | None:
        """Resolve an f-string in an Enum __init__ using the enum member values.

        Handles patterns like: f"codelists/{self.codelist_name}.csv"

        Args:
            fstring: AST JoinedStr node representing the f-string
            enum_member_values: The tuple values from the enum member

        Returns:
            Resolved string or None if cannot resolve
        """
        parts: list[str] = []
        for value in fstring.values:
            if isinstance(value, ast.Constant):
                # Static part of the f-string
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                # Dynamic part - check if it's self.codelist_name
                if isinstance(value.value, ast.Attribute):
                    if (
                        isinstance(value.value.value, ast.Name)
                        and value.value.value.id == "self"
                    ):
                        attr = value.value.attr
                        # Map self.codelist_name to the first enum value
                        if attr == "codelist_name" and len(enum_member_values) > 0:
                            parts.append(enum_member_values[0])
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            else:
                return None

        return "".join(parts) if parts else None

    def _trace_name(
        self,
        name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Trace a Name reference to find codelists.

        Args:
            name: Variable name to trace
            tree: AST tree of current file
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        # Check cache
        cache_key = (str(file_path), name)
        if cache_key in self._codelist_cache:
            return self._codelist_cache[cache_key]

        # Prevent cycles
        if cache_key in self._visited_vars:
            return []
        self._visited_vars.add(cache_key)

        codelist_calls: list[CodelistCall] = []

        # Check if this is an imported name
        if name in import_collector.imported_modules:
            module_name, original_name = import_collector.imported_modules[name]
            target_name = original_name or name
            calls = self._trace_imported_name(
                module_name, target_name, import_collector, depth
            )
            codelist_calls.extend(calls)
        else:
            # Look for local variable definition using index
            calls = self._find_local_definition(
                name, tree, import_collector, file_path, ast_index, depth
            )
            codelist_calls.extend(calls)

        # Cache result
        self._codelist_cache[cache_key] = codelist_calls
        return codelist_calls

    def _trace_attribute(
        self,
        attr_node: ast.Attribute,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Trace an Attribute access to find codelists.

        Handles patterns like:
        - Codelists.DIABETES.codes
        - module.codelist_var
        - obj.attribute

        Args:
            attr_node: AST Attribute node
            tree: AST tree of current file
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        codelist_calls: list[CodelistCall] = []

        # Get the object being accessed
        if isinstance(attr_node.value, ast.Name):
            obj_name = attr_node.value.id
            attr_name = attr_node.attr

            # Check if obj_name is an imported class/module
            if obj_name in import_collector.imported_modules:
                module_name, original_name = import_collector.imported_modules[obj_name]

                # Check if this is a simple module import (import codelists)
                # vs a from-import (from codelists import X)
                if original_name is None:
                    # This is "import module_name" so trace the attribute in that module
                    calls = self._trace_imported_name(
                        module_name, attr_name, import_collector, depth
                    )
                    codelist_calls.extend(calls)
                else:
                    # This is "from module_name import original_name as obj_name"
                    # Treat as class attribute access
                    target_class = original_name
                    calls = self._trace_class_attribute(
                        module_name, target_class, attr_name, import_collector, depth
                    )
                    codelist_calls.extend(calls)
            # Check if this is a dataset variable reference (e.g., dataset.ppi)
            elif obj_name == "dataset":
                # Find the dataset variable assignment and trace its expression using index
                calls = self._find_dataset_variable_reference(
                    attr_name, tree, import_collector, file_path, ast_index, depth
                )
                codelist_calls.extend(calls)
            else:
                # Check if it's a local class
                calls = self._trace_local_class_attribute(
                    obj_name,
                    attr_name,
                    tree,
                    import_collector,
                    file_path,
                    ast_index,
                    depth,
                )
                codelist_calls.extend(calls)

        # Handle nested attributes like obj.attr1.attr2
        elif isinstance(attr_node.value, ast.Attribute):
            calls = self._trace_attribute(
                attr_node.value, tree, import_collector, file_path, ast_index, depth
            )
            codelist_calls.extend(calls)

        return codelist_calls

    def _find_dataset_variable_reference(
        self,
        var_name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Find a dataset variable assignment and trace its expression.

        When we see dataset.var_name in an expression, find where it was assigned
        (dataset.var_name = expr) and trace that expression.

        Args:
            var_name: The dataset variable name (e.g., "ppi")
            tree: AST tree to search
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        # Prevent infinite recursion
        if depth > 50:
            return []

        codelist_calls: list[CodelistCall] = []

        # Use index for O(1) lookup
        assign_node = ast_index.dataset_assignments.get(var_name)
        if assign_node:
            # Found the assignment, trace the right-hand side
            calls = self._trace_expression(
                assign_node.value,
                tree,
                import_collector,
                file_path,
                ast_index,
                depth + 1,
            )
            codelist_calls.extend(calls)

        return codelist_calls

    def _find_local_definition(
        self,
        var_name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Find and trace a local variable definition.

        Args:
            var_name: Variable name to find
            tree: AST tree to search
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        codelist_calls: list[CodelistCall] = []

        # Use index for O(1) lookup instead of O(n) walk
        assign_nodes = ast_index.var_assignments.get(var_name, [])

        for node in assign_nodes:
            # Recursively trace the RHS
            calls = self._trace_expression(
                node.value, tree, import_collector, file_path, ast_index, depth + 1
            )
            codelist_calls.extend(calls)

        if codelist_calls:
            return codelist_calls

        function_defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == var_name
        ]
        for function_def in function_defs:
            for return_node in self._iter_non_nested_returns(function_def):
                if return_node.value is None:
                    continue
                calls = self._trace_expression(
                    return_node.value,
                    tree,
                    import_collector,
                    file_path,
                    ast_index,
                    depth + 1,
                )
                codelist_calls.extend(calls)

        return codelist_calls

    def _iter_non_nested_returns(self, func_def: ast.FunctionDef) -> list[ast.Return]:
        returns: list[ast.Return] = []

        def visit_statements(statements: list[ast.stmt]) -> None:
            for stmt in statements:
                if isinstance(stmt, ast.Return):
                    returns.append(stmt)
                    continue

                if isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue

                if isinstance(stmt, ast.If):
                    visit_statements(stmt.body)
                    visit_statements(stmt.orelse)
                    continue

                if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                    visit_statements(stmt.body)
                    visit_statements(stmt.orelse)
                    continue

                if isinstance(stmt, (ast.With, ast.AsyncWith)):
                    visit_statements(stmt.body)
                    continue

                if isinstance(stmt, ast.Try):
                    visit_statements(stmt.body)
                    visit_statements(stmt.orelse)
                    visit_statements(stmt.finalbody)
                    for handler in stmt.handlers:
                        visit_statements(handler.body)
                    continue

                if hasattr(ast, "Match") and isinstance(stmt, ast.Match):
                    for case_node in stmt.cases:
                        visit_statements(case_node.body)

        visit_statements(func_def.body)
        return returns

    def _trace_imported_name(
        self,
        module_name: str,
        target_name: str,
        import_collector: ImportCollector,
        depth: int,
    ) -> list[CodelistCall]:
        """Trace an imported name to find codelists in the source module.

        Args:
            module_name: Module to search
            target_name: Name of the variable/function in that module
            import_collector: Import information
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                module_file_str = str(module_file)

                # Check cache first
                if module_file_str in self._module_tree_cache:
                    module_tree, module_import_collector, module_index = (
                        self._module_tree_cache[module_file_str]
                    )
                else:
                    # Parse and cache
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=module_file_str)

                    # Create import collector for this module
                    module_import_collector = ImportCollector()
                    module_import_collector.collect(module_tree)
                    module_import_collector.resolve_star_imports(self.module_resolver)

                    # Create index for this module
                    module_index = ASTIndex(module_tree)

                    # Cache it
                    self._module_tree_cache[module_file_str] = (
                        module_tree,
                        module_import_collector,
                        module_index,
                    )

                # Find the definition in this module
                return self._find_local_definition(
                    target_name,
                    module_tree,
                    module_import_collector,
                    module_file,
                    module_index,
                    depth + 1,
                )
            except Exception:
                continue

        return []

    def _trace_class_attribute(
        self,
        module_name: str,
        class_name: str,
        attr_name: str,
        import_collector: ImportCollector,
        depth: int,
    ) -> list[CodelistCall]:
        """Trace a class attribute to find codelists.

        Handles patterns like: Codelists.DIABETES where Codelists is a class.

        Args:
            module_name: Module containing the class
            class_name: Name of the class
            attr_name: Attribute name to find
            import_collector: Import information
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                module_file_str = str(module_file)

                # Check cache first
                if module_file_str in self._module_tree_cache:
                    module_tree, module_import_collector, module_index = (
                        self._module_tree_cache[module_file_str]
                    )
                else:
                    # Parse and cache
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=module_file_str)

                    # Create import collector for this module
                    module_import_collector = ImportCollector()
                    module_import_collector.collect(module_tree)
                    module_import_collector.resolve_star_imports(self.module_resolver)

                    # Create index for this module
                    module_index = ASTIndex(module_tree)

                    # Cache it
                    self._module_tree_cache[module_file_str] = (
                        module_tree,
                        module_import_collector,
                        module_index,
                    )

                # Use index for O(1) class lookup
                class_def = module_index.class_defs.get(class_name)
                if class_def:
                    return self._find_class_attribute_definition(
                        class_def,
                        attr_name,
                        module_tree,
                        module_import_collector,
                        module_file,
                        module_index,
                        depth,
                    )

            except Exception:
                continue

        return []

    def _find_class_attribute_definition(
        self,
        class_node: ast.ClassDef,
        attr_name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Find an attribute definition within a class.

        Args:
            class_node: AST ClassDef node
            attr_name: Attribute name to find
            tree: Full AST tree
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        codelist_calls: list[CodelistCall] = []

        # Check if this is an Enum class
        is_enum = False
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == "Enum":
                is_enum = True
                break

        if is_enum:
            # For Enum classes, extract the enum member value and resolve codelist calls
            # The pattern is: MEMBER_NAME = (value1, value2, ...)
            # And __init__ has: self.codes = codelist_from_csv(f"codelists/{self.codelist_name}.csv", ...)

            # First, find the enum member's value tuple
            member_values: tuple[str, ...] | None = None
            for node in class_node.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            # Extract the tuple value
                            if isinstance(node.value, ast.Tuple):
                                values: list[str] = []
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        values.append(str(elt.value))
                                if values:
                                    member_values = tuple(values)
                            break

            if not member_values:
                return []

            # Now find the __init__ and extract codelist_from_csv calls
            # We'll need to resolve any f-strings using the member_values
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    # Find all codelist_from_csv calls in __init__
                    for init_node in ast.walk(node):
                        if isinstance(init_node, ast.Call):
                            calls = self._check_for_codelist_call_in_enum(
                                init_node, member_values
                            )
                            codelist_calls.extend(calls)
                    # Don't need to check further if we found calls in __init__
                    if codelist_calls:
                        return codelist_calls

        # Check class-level assignments (for non-Enum classes)
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        calls = self._trace_expression(
                            node.value,
                            tree,
                            import_collector,
                            file_path,
                            ast_index,
                            depth + 1,
                        )
                        codelist_calls.extend(calls)

        return codelist_calls

    def _trace_local_class_attribute(
        self,
        class_name: str,
        attr_name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        ast_index: ASTIndex,
        depth: int,
    ) -> list[CodelistCall]:
        """Trace an attribute of a locally-defined class or instance.

        Args:
            class_name: Name of the local class OR variable that's an instance
            attr_name: Attribute name to find
            tree: AST tree
            import_collector: Import information
            file_path: Current file path
            ast_index: Pre-computed AST index
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        # First check if class_name is a variable assigned to a class instance
        if class_name in ast_index.var_assignments:
            for assign in ast_index.var_assignments[class_name]:
                # Check if it's assigned from a class instantiation
                if isinstance(assign.value, ast.Call) and isinstance(
                    assign.value.func, ast.Name
                ):
                    actual_class_name = assign.value.func.id
                    class_def = ast_index.class_defs.get(actual_class_name)
                    if class_def:
                        return self._find_class_attribute_definition(
                            class_def,
                            attr_name,
                            tree,
                            import_collector,
                            file_path,
                            ast_index,
                            depth,
                        )

        # Use index for O(1) class lookup
        class_def = ast_index.class_defs.get(class_name)
        if class_def:
            return self._find_class_attribute_definition(
                class_def,
                attr_name,
                tree,
                import_collector,
                file_path,
                ast_index,
                depth,
            )

        return []


class VariableExtractor:
    """Main class for extracting variable definitions from ehrQL files."""

    def __init__(self, file_path: pathlib.Path, repo_root: pathlib.Path):
        self.file_path = file_path
        self.repo_root = repo_root
        self.module_resolver = ModuleResolver(file_path, repo_root)
        self.name_extractor = DynamicNameExtractor()
        self.operation_finder = DatasetOperationFinder(self.name_extractor)
        self.function_analyzer = FunctionAnalyzer(self.operation_finder)
        self.codelist_tracer = CodelistTracer(self.module_resolver)

    def extract(
        self,
    ) -> tuple[
        dict[str, int | tuple[str, int]], list[tuple[str, int | tuple[str, int]]]
    ]:
        """Extract all variable definitions from the file.

        Returns:
            Tuple of:
            - dict mapping variable_name -> line_number (int) or (filename, line_number)
            - list of (regex_pattern, line_number_or_tuple) for dynamic variables
              where line_number_or_tuple is int for same-file or (filename, line) for cross-file
        """
        line_numbers: dict[str, int | tuple[str, int]] = {}
        line_number_regexes: list[tuple[str, int | tuple[str, int]]] = []

        try:
            with open(self.file_path, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(self.file_path))
        except Exception:
            return {}, []

        # Collect imports and function definitions
        import_collector = ImportCollector()
        import_collector.collect(tree)
        import_collector.resolve_star_imports(self.module_resolver)

        # Pass 1: Find dataset = create_function() and extract from that function
        self._extract_from_dataset_creator(
            tree, import_collector, line_numbers, line_number_regexes
        )

        # Pass 2: Direct module-level dataset operations
        self._extract_module_level(tree, line_numbers, line_number_regexes)

        # Pass 3: Loop-based dynamic variables
        self._extract_from_loops(
            tree, import_collector, line_numbers, line_number_regexes
        )

        # Pass 4: Non-loop helper function calls
        self._extract_from_helpers(
            tree, import_collector, line_numbers, line_number_regexes
        )

        return line_numbers, line_number_regexes

    def _extract_from_dataset_creator(
        self,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract variables from dataset = create_function() patterns."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == "dataset"):
                    continue

                if not isinstance(node.value, ast.Call):
                    continue

                if not isinstance(node.value.func, ast.Name):
                    continue

                func_name = node.value.func.id

                if func_name == "create_dataset":
                    continue

                # Try local function first
                if func_name in import_collector.function_defs:
                    func_def = import_collector.function_defs[func_name]
                    self._extract_from_local_function(
                        func_def, line_numbers, line_number_regexes
                    )

                # Try imported function
                elif func_name in import_collector.imported_modules:
                    module_name, original_func_name = import_collector.imported_modules[
                        func_name
                    ]
                    target_name = original_func_name or func_name
                    self._extract_from_imported_function(
                        module_name, target_name, line_numbers, line_number_regexes
                    )

    def _extract_from_local_function(
        self,
        func_def: ast.FunctionDef,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract from a function defined in the same file."""
        static_vars, dynamic_patterns = self.function_analyzer.extract_from_function(
            func_def, "dataset"
        )

        for var_name, line in static_vars:
            line_numbers[var_name] = line

        for pattern, line in dynamic_patterns:
            line_number_regexes.append((pattern, line))

        # Handle dict-return pattern: variables = generate_variables(...); for k,v in variables.items(): setattr(dataset, k, v)
        for key, source_file, def_line in self._extract_dict_setattr_from_function(
            func_def
        ):
            # Only set if not already present
            if key not in line_numbers:
                if source_file:
                    # Cross-file reference (even from a local function calling imported helper)
                    line_numbers[key] = (source_file, def_line)
                else:
                    # Same file
                    line_numbers[key] = def_line

    def _extract_from_imported_function(
        self,
        module_name: str,
        func_name: str,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract from a function defined in an imported module."""
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                for node in ast.walk(module_tree):
                    if not (
                        isinstance(node, ast.FunctionDef) and node.name == func_name
                    ):
                        continue

                    static_vars, dynamic_patterns = (
                        self.function_analyzer.extract_from_function(node, "dataset")
                    )

                    rel_path = self.module_resolver.get_relative_path(module_file)

                    for var_name, line in static_vars:
                        line_numbers[var_name] = (rel_path, line)

                    for pattern, line in dynamic_patterns:
                        line_number_regexes.append((pattern, line))

                    # Handle dict-return setattr loop pattern inside imported function
                    for (
                        key,
                        source_file,
                        def_line,
                    ) in self._extract_dict_setattr_from_function(node):
                        if key not in line_numbers:
                            # If source_file is provided, it's a cross-file reference to variables file
                            if source_file:
                                line_numbers[key] = (source_file, def_line)
                            else:
                                # Same file reference
                                line_numbers[key] = (rel_path, def_line)

                    return  # Found the function
            except Exception:
                continue

    def _extract_dict_setattr_from_function(
        self, func_def: ast.FunctionDef
    ) -> list[tuple[str, str | None, int]]:
        """Detect pattern where a function builds a dict of variables and then assigns via setattr.

        Looks for:
            vars = some_function(...)
            for key, value in vars.items():
                setattr(dataset, key, value)

        Returns a list of (key_name, source_file_or_none, definition_line) tuples.
        If source_file is None, the definition is in the same file as func_def.
        If source_file is set, it's a relative path to the file where the variable is defined.
        """
        # Collect inline imports inside the function (to resolve from X import Y)
        from_imports: dict[str, str] = {}
        import_aliases: dict[str, str] = {}
        for node in ast.walk(func_def):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    from_imports[name] = module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    import_aliases[name] = alias.name

        # Map variable name to (module_name, function_name) when assigned from a call
        var_sources: dict[str, tuple[str | None, str]] = {}
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                if (
                    len(node.targets) >= 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                ):
                    target_name = node.targets[0].id
                    callee = node.value.func
                    module_name: str | None = None
                    func_name: str | None = None
                    if isinstance(callee, ast.Name):
                        func_name = callee.id
                        module_name = from_imports.get(func_name)
                    elif isinstance(callee, ast.Attribute) and isinstance(
                        callee.value, ast.Name
                    ):
                        mod_alias = callee.value.id
                        func_name = callee.attr
                        module_name = import_aliases.get(mod_alias, mod_alias)
                    if func_name:
                        var_sources[target_name] = (module_name, func_name)

        results: list[tuple[str, str | None, int]] = []
        # Scan for loops over dict.items()
        for node in ast.walk(func_def):
            if not isinstance(node, ast.For):
                continue
            # iter must be something.items()
            dict_var_name: str | None = None
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"
                and isinstance(node.iter.func.value, ast.Name)
            ):
                dict_var_name = node.iter.func.value.id
            if not dict_var_name:
                continue

            # The key variable name in for key, value in ...
            key_var_name: str | None = None
            if (
                isinstance(node.target, ast.Tuple)
                and len(node.target.elts) >= 1
                and isinstance(node.target.elts[0], ast.Name)
            ):
                key_var_name = node.target.elts[0].id
            elif isinstance(node.target, ast.Name):
                key_var_name = node.target.id
            if not key_var_name:
                continue

            # Ensure the body contains setattr(dataset, key_var_name, ...)
            found_setattr = False
            for bn in ast.walk(node):
                if (
                    isinstance(bn, ast.Call)
                    and isinstance(bn.func, ast.Name)
                    and bn.func.id == "setattr"
                ):
                    if (
                        len(bn.args) >= 2
                        and isinstance(bn.args[0], ast.Name)
                        and bn.args[0].id == "dataset"
                        and isinstance(bn.args[1], ast.Name)
                        and bn.args[1].id == key_var_name
                    ):
                        found_setattr = True
                        break
            if not found_setattr:
                continue

            src = var_sources.get(dict_var_name)
            if not src:
                continue
            module_name, func_name = src
            key_lines = self._resolve_returned_dict_keys_with_lines(
                module_name, func_name
            )
            for k, def_line in key_lines:
                # Determine source file
                source_file: str | None = None
                if module_name:
                    # Cross-file reference
                    for candidate_file in self.module_resolver.find_module_file(
                        module_name
                    ):
                        if candidate_file.exists():
                            source_file = self.module_resolver.get_relative_path(
                                candidate_file
                            )
                            break
                results.append((k, source_file, def_line))

        return results

    def _resolve_returned_dict_keys_with_lines(
        self, module_name: str | None, func_name: str
    ) -> list[tuple[str, int]]:
        """Resolve function returning a dict of variables and return (key, definition_line) pairs.

        Supports:
        - Returning a dict literal: return {"a": expr, ...}
        - Returning a variable that was assigned a dict literal earlier in the function
        - Returning a dict constructed via dict(key=value, ...) either directly or via a named variable

        For dict(cov_bin_ckd=cov_bin_ckd, ...), returns the line where cov_bin_ckd (the RHS) is defined.
        """
        if not func_name:
            return []
        # Build candidate files
        candidate_files: list[pathlib.Path] = []
        if module_name:
            candidate_files = self.module_resolver.find_module_file(module_name)
        else:
            candidate_files = [self.file_path]

        for module_file in candidate_files:
            if not module_file.exists():
                continue
            try:
                with open(module_file, encoding="utf-8") as f:
                    src = f.read()
                tree = ast.parse(src, filename=str(module_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        # Helper: extract (key, line) pairs from a dict expr or dict(...) call
                        def extract_keys_with_lines(
                            expr: ast.AST,
                        ) -> list[tuple[str, int]]:
                            results: list[tuple[str, int]] = []
                            if isinstance(expr, ast.Dict):
                                # Dict literal: {"key": value, ...}
                                for k in expr.keys:
                                    if isinstance(k, ast.Constant) and isinstance(
                                        k.value, str
                                    ):
                                        # Use the line of the dict itself as fallback
                                        results.append(
                                            (k.value, getattr(expr, "lineno", 0))
                                        )
                            elif isinstance(expr, ast.Call):
                                # dict(key=value, ...) form
                                callee = expr.func
                                if isinstance(callee, ast.Name) and callee.id == "dict":
                                    # For dict(cov_bin_ckd=cov_bin_ckd, ...), we need to find where
                                    # the RHS variable (cov_bin_ckd) is defined
                                    for kw in expr.keywords:
                                        if kw.arg is not None:
                                            key_name = kw.arg
                                            # Check if the value is a simple Name reference
                                            if isinstance(kw.value, ast.Name):
                                                rhs_var = kw.value.id
                                                # Find where this variable is assigned in the function
                                                def_line = find_variable_definition(
                                                    node, rhs_var
                                                )
                                                if def_line:
                                                    results.append((key_name, def_line))
                                                else:
                                                    # Fallback to the dict(...) call line
                                                    results.append(
                                                        (
                                                            key_name,
                                                            getattr(expr, "lineno", 0),
                                                        )
                                                    )
                                            else:
                                                # Value is an expression, use the dict(...) line
                                                results.append(
                                                    (
                                                        key_name,
                                                        getattr(expr, "lineno", 0),
                                                    )
                                                )
                            return results

                        def find_variable_definition(
                            func_node: ast.FunctionDef, var_name: str
                        ) -> int | None:
                            """Find the line where a variable is assigned in a function."""
                            candidates: list[int] = []
                            for sub in ast.walk(func_node):
                                if isinstance(sub, ast.Assign):
                                    for t in sub.targets:
                                        if isinstance(t, ast.Name) and t.id == var_name:
                                            candidates.append(getattr(sub, "lineno", 0))
                            if candidates:
                                return max(candidates)  # Return the last assignment
                            return None

                        # Case 1: direct return of dict literal or dict(...) call
                        for ret in ast.walk(node):
                            if isinstance(ret, ast.Return) and ret.value is not None:
                                direct_results = extract_keys_with_lines(ret.value)
                                if direct_results:
                                    return direct_results
                                # Case 2: return a name that was assigned to a dict/dict(...)
                                if isinstance(ret.value, ast.Name):
                                    var_name = ret.value.id
                                    # Find assignments to this name within the function
                                    candidates: list[tuple[int, ast.AST]] = []
                                    for sub in ast.walk(node):
                                        if isinstance(sub, ast.Assign):
                                            target_names: list[str] = []
                                            for t in sub.targets:
                                                if isinstance(t, ast.Name):
                                                    target_names.append(t.id)
                                            if var_name in target_names:
                                                candidates.append(
                                                    (
                                                        getattr(sub, "lineno", 0),
                                                        sub.value,
                                                    )
                                                )
                                    if candidates:
                                        # choose the assignment with max lineno (last in source)
                                        _, expr = max(candidates, key=lambda x: x[0])
                                        results = extract_keys_with_lines(expr)
                                        if results:
                                            return results
            except Exception:
                continue
        return []

    def _extract_module_level(
        self,
        tree: ast.AST,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract direct module-level dataset operations.

        Only processes nodes at module level, not inside function definitions.
        """
        for node in ast.iter_child_nodes(tree):
            # Skip function definitions - they're handled separately
            if isinstance(node, ast.FunctionDef):
                continue

            # Walk children of this node
            for child in ast.walk(node):
                # Attribute assignments
                for var_name, line in self.operation_finder.find_variable_assignments(
                    child
                ):
                    if var_name not in line_numbers:
                        line_numbers[var_name] = line

                # add_column calls
                static, dynamic = self.operation_finder.find_add_column_calls(child)
                for var_name, line in static:
                    if var_name not in line_numbers:
                        line_numbers[var_name] = line
                for pattern, line in dynamic:
                    line_number_regexes.append((pattern, line))

    def _extract_from_loops(
        self,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract dynamic variables from loops."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.For, ast.While)):
                self._process_loop(
                    node, tree, import_collector, line_numbers, line_number_regexes
                )

    def _process_loop(
        self,
        loop_node: ast.For | ast.While,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Process a single loop node to find dynamic variables."""
        loop_line = loop_node.lineno

        # Check if this is a dict.items() loop pattern
        if self._extract_from_dict_items_loop(
            loop_node, tree, import_collector, line_numbers
        ):
            # Successfully extracted from dict.items() pattern, nothing more to do
            return

        for child in ast.walk(loop_node):
            # Direct dataset operations in loop
            static, dynamic = self.operation_finder.find_add_column_calls(child)
            # Static variables with Name nodes (e.g., loop variables) that need resolution
            for var_name, _ in static:
                # Try to resolve the variable name from loop context
                if isinstance(loop_node, ast.For) and isinstance(
                    loop_node.target, ast.Tuple
                ):
                    # Check if this is the loop variable (e.g., "comorb" in for comorb, val in dict.items())
                    # If so, we need dict.items() pattern handling which is done above
                    pass
            for pattern, _ in dynamic:
                line_number_regexes.append((pattern, loop_line))

            static, dynamic = self.operation_finder.find_setattr_calls(child)
            for pattern, _ in dynamic:
                line_number_regexes.append((pattern, loop_line))

            static, dynamic = self.operation_finder.find_subscript_assignments(child)
            for pattern, _ in dynamic:
                line_number_regexes.append((pattern, loop_line))

            # Function calls in loop
            if isinstance(child, ast.Call):
                self._process_loop_function_call(
                    child, import_collector, loop_line, line_number_regexes
                )

    def _extract_from_dict_items_loop(
        self,
        loop_node: ast.For | ast.While,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
    ) -> bool:
        """Extract variables from a dict.items() loop pattern.

        Pattern 1 (imported dict with setattr):
            for var_name, var_value in imported_dict.items():
                setattr(dataset, var_name, var_value)

        Pattern 2 (local dict with add_column):
            my_dict = {"key1": val1, "key2": val2}
            for key, value in my_dict.items():
                dataset.add_column(key, value)

        Returns True if this pattern was detected and processed.
        """
        # Only handle For loops
        if not isinstance(loop_node, ast.For):
            return False

        # Check if iterating over a .items() call
        if not (
            isinstance(loop_node.iter, ast.Call)
            and isinstance(loop_node.iter.func, ast.Attribute)
            and loop_node.iter.func.attr == "items"
        ):
            return False

        # Get the dict name
        if not isinstance(loop_node.iter.func.value, ast.Name):
            return False

        dict_name = loop_node.iter.func.value.id

        # Try Pattern 1: Imported dict with setattr
        if dict_name in import_collector.imported_modules:
            module_name, original_name = import_collector.imported_modules[dict_name]
            target_dict_name = original_name or dict_name

            # Find the setattr call in the loop body
            setattr_node = None
            for node in ast.walk(loop_node):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                ):
                    setattr_node = node
                    break

            if setattr_node and len(setattr_node.args) >= 3:
                # Now we need to find the dict definition in the imported module
                # and extract the variable names and their definition line numbers
                module_candidates = self.module_resolver.find_module_file(module_name)
                module_path = None
                for candidate in module_candidates:
                    if candidate.exists():
                        module_path = candidate
                        break

                if module_path:
                    try:
                        with open(module_path) as f:
                            module_source = f.read()
                        module_tree = ast.parse(module_source)
                    except (OSError, SyntaxError):
                        return False

                    # Find the dict assignment in the module
                    dict_definition = self._find_dict_definition(
                        module_tree, target_dict_name
                    )
                    if dict_definition:
                        # Extract variable definitions from the dict
                        rel_path = str(
                            pathlib.Path(module_path).relative_to(self.repo_root)
                        )
                        for key, value_node in dict_definition.items():
                            # key is the variable name (e.g., "cens_date_death")
                            # value_node is the AST node for the value (e.g., Name("death_date"))

                            # If the value is a Name node, find where that variable was defined
                            if isinstance(value_node, ast.Name):
                                var_def_line = self._find_variable_definition_line(
                                    module_tree, value_node.id
                                )
                                if var_def_line:
                                    line_numbers[key] = (rel_path, var_def_line)

                        return True

        # Try Pattern 2: Local dict with add_column
        # Find add_column call in the loop body
        add_column_node = None
        for node in ast.walk(loop_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_column"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dataset"
            ):
                add_column_node = node
                break

        if add_column_node and len(add_column_node.args) >= 1:
            # Find the local dict definition
            dict_dict, dict_line = self._find_local_dict_literal(tree, dict_name)
            if dict_dict:
                # Extract keys from the dict literal
                for key, key_line in dict_dict.items():
                    line_numbers[key] = key_line
                return True

        return False

    def _find_dict_definition(
        self, tree: ast.AST, dict_name: str
    ) -> dict[str, ast.AST] | None:
        """Find a dict definition and return its keys and values.

        Looks for patterns like:
            my_dict = dict(key1=value1, key2=value2)
            my_dict = {"key1": value1, "key2": value2}
        """
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == dict_name
            ):
                # Check if RHS is a dict() call
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"
                ):
                    # Extract keyword arguments as dict entries
                    result = {}
                    for keyword in node.value.keywords:
                        if keyword.arg:  # keyword.arg is the key name
                            result[keyword.arg] = keyword.value
                    return result
                # Check if RHS is a dict literal
                elif isinstance(node.value, ast.Dict):
                    result = {}
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        # Only handle string literal keys
                        if isinstance(key_node, ast.Constant) and isinstance(
                            key_node.value, str
                        ):
                            result[key_node.value] = value_node
                    return result if result else None

        return None

    def _find_list_literal_values(
        self, tree: ast.AST, list_name: str
    ) -> list[str] | None:
        """Find a list definition and return its string values.

        Looks for patterns like:
            my_list = ["value1", "value2", "value3"]
        """
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == list_name
            ):
                # Check if RHS is a list literal
                if isinstance(node.value, ast.List):
                    result = []
                    for elt in node.value.elts:
                        # Only handle string literal values
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            result.append(elt.value)
                    return result if result else None

        return None

    def _find_variable_definition_line(
        self, tree: ast.AST, var_name: str
    ) -> int | None:
        """Find the line number where a variable is defined."""
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == var_name
            ):
                return node.lineno

        return None

    def _find_local_dict_literal(
        self, tree: ast.AST, dict_name: str
    ) -> tuple[dict[str, int], int] | tuple[None, None]:
        """Find a local dict literal and return its keys with line numbers.

        Looks for patterns like:
            my_dict = {
                "key1": value1,  # line 5
                "key2": value2,  # line 6
            }

        Returns:
            Tuple of (dict_mapping, dict_line) where:
            - dict_mapping: dict[key_string -> line_number]
            - dict_line: line number of the dict assignment
            Returns (None, None) if not found
        """
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == dict_name
                and isinstance(node.value, ast.Dict)
            ):
                # Extract keys and their line numbers
                result = {}
                for key_node in node.value.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(
                        key_node.value, str
                    ):
                        # Use the line number of the key node
                        result[key_node.value] = key_node.lineno

                return result, node.lineno

        return None, None

    def _process_loop_function_call(
        self,
        call_node: ast.Call,
        import_collector: ImportCollector,
        loop_line: int,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Process a function call inside a loop."""
        # Handle direct function name call
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id

            # Local function
            if func_name in import_collector.function_defs:
                func_def = import_collector.function_defs[func_name]
                self._extract_from_loop_helper(
                    func_def, call_node, loop_line, line_number_regexes
                )

            # Imported function
            elif func_name in import_collector.imported_modules:
                module_name, original_name = import_collector.imported_modules[
                    func_name
                ]
                target_name = original_name or func_name
                self._extract_from_imported_loop_helper(
                    module_name, target_name, call_node, loop_line, line_number_regexes
                )

        # Handle module.function() call
        elif isinstance(call_node.func, ast.Attribute) and isinstance(
            call_node.func.value, ast.Name
        ):
            mod_alias = call_node.func.value.id
            func_name = call_node.func.attr

            if mod_alias in import_collector.imported_modules:
                module_name, _ = import_collector.imported_modules[mod_alias]
                self._extract_from_imported_loop_helper(
                    module_name, func_name, call_node, loop_line, line_number_regexes
                )

    def _extract_from_loop_helper(
        # End of public API
        self,
        func_def: ast.FunctionDef,
        call_node: ast.Call,
        loop_line: int,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract patterns from a helper function called in a loop.

        Returns the line numbers from inside the helper function, not the loop line.
        """
        ds_idx = self.function_analyzer.find_dataset_param_index(func_def, call_node)
        if ds_idx is None or ds_idx >= len(func_def.args.args):
            return

        dataset_param_name = func_def.args.args[ds_idx].arg
        _, dynamic_patterns = self.function_analyzer.extract_from_function(
            func_def, dataset_param_name
        )

        # Use the actual line from inside the helper, not the loop line
        for pattern, helper_line in dynamic_patterns:
            line_number_regexes.append((pattern, helper_line))

    def _extract_from_imported_loop_helper(
        self,
        module_name: str,
        func_name: str,
        call_node: ast.Call,
        loop_line: int,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract patterns from an imported helper function called in a loop.

        Returns tuples with the source filename for cross-file patterns.
        """
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                for node in ast.walk(module_tree):
                    if not (
                        isinstance(node, ast.FunctionDef) and node.name == func_name
                    ):
                        continue

                    # Extract patterns from the helper
                    ds_idx = self.function_analyzer.find_dataset_param_index(
                        node, call_node
                    )
                    if ds_idx is None or ds_idx >= len(node.args.args):
                        return

                    dataset_param_name = node.args.args[ds_idx].arg
                    _, dynamic_patterns = self.function_analyzer.extract_from_function(
                        node, dataset_param_name
                    )

                    # Return cross-file tuples with the helper file and line
                    rel_path = self.module_resolver.get_relative_path(module_file)
                    for pattern, helper_line in dynamic_patterns:
                        line_number_regexes.append((pattern, (rel_path, helper_line)))

                    return
            except Exception:
                continue

    def _extract_from_helpers(
        self,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract from non-loop helper function calls."""
        # First pass: collect instance variables and their classes
        instance_classes: dict[
            str, tuple[str, str]
        ] = {}  # instance_name -> (module_name, class_name)

        # Also track constructor call nodes for resolving instance attributes
        instance_constructor_calls: dict[
            str, ast.Call
        ] = {}  # instance_name -> constructor Call node

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for: instance = ClassName(...)
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(
                        node.value, ast.Call
                    ):
                        instance_name = target.id

                        if isinstance(node.value.func, ast.Name):
                            class_name = node.value.func.id
                            # Check if this is an imported class
                            if class_name in import_collector.imported_modules:
                                module_name, original_name = (
                                    import_collector.imported_modules[class_name]
                                )
                                actual_class_name = original_name or class_name
                                instance_classes[instance_name] = (
                                    module_name,
                                    actual_class_name,
                                )
                                # Store the constructor call node
                                instance_constructor_calls[instance_name] = node.value

        for node in ast.iter_child_nodes(tree):
            # Skip loops
            if isinstance(node, (ast.For, ast.While)):
                continue

            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue

                call_line = child.lineno

                # Direct function call
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id

                    # Local function
                    if func_name in import_collector.function_defs:
                        func_def = import_collector.function_defs[func_name]
                        self._extract_from_standalone_helper(
                            func_def,
                            child,
                            call_line,
                            line_numbers,
                            line_number_regexes,
                        )

                    # Imported function
                    elif func_name in import_collector.imported_modules:
                        module_name, original_name = import_collector.imported_modules[
                            func_name
                        ]
                        target_name = original_name or func_name
                        static_vars = self._extract_from_imported_standalone_helper(
                            module_name,
                            target_name,
                            child,
                            call_line,
                            line_number_regexes,
                        )
                        # Add static vars from setattr-with-param pattern
                        for var_name, line in static_vars:
                            line_numbers[var_name] = line

                # Module.function() call OR instance.method() call
                elif isinstance(child.func, ast.Attribute) and isinstance(
                    child.func.value, ast.Name
                ):
                    obj_name = child.func.value.id
                    method_or_func_name = child.func.attr

                    # Check if this is an instance method call
                    if obj_name in instance_classes:
                        module_name, class_name = instance_classes[obj_name]
                        constructor_call = instance_constructor_calls.get(obj_name)
                        self._extract_from_class_method(
                            module_name,
                            class_name,
                            method_or_func_name,
                            child,
                            constructor_call,
                            line_numbers,
                            line_number_regexes,
                        )
                    # Otherwise, check if it's a module.function() call
                    elif obj_name in import_collector.imported_modules:
                        module_name, _ = import_collector.imported_modules[obj_name]
                        static_vars = self._extract_from_imported_standalone_helper(
                            module_name,
                            method_or_func_name,
                            child,
                            call_line,
                            line_number_regexes,
                        )
                        # Add static vars from setattr-with-param pattern
                        for var_name, line in static_vars:
                            line_numbers[var_name] = line

    def _extract_from_standalone_helper(
        self,
        func_def: ast.FunctionDef,
        call_node: ast.Call,
        call_line: int,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract from helper function called outside a loop.

        Returns the line numbers from inside the helper function, not the call line.
        """
        ds_idx = self.function_analyzer.find_dataset_param_index(func_def, call_node)

        # If dataset is a parameter, use the parameter name
        # Otherwise, assume it's a global variable named "dataset"
        if ds_idx is not None and ds_idx < len(func_def.args.args):
            dataset_param_name = func_def.args.args[ds_idx].arg
        else:
            dataset_param_name = "dataset"

        static_vars, dynamic_patterns = self.function_analyzer.extract_from_function(
            func_def, dataset_param_name
        )

        # For static variables that are parameters, resolve them from call arguments
        # e.g., if function has dataset.add_column(column_name, ...) and column_name
        # is a parameter, get the actual string from the call site
        for var_name, helper_line in static_vars:
            # Try to resolve the parameter to an actual argument value
            actual_var_name = self._resolve_param_to_arg(func_def, call_node, var_name)

            # If we resolved it, use the resolved name
            # If we couldn't resolve it (not a parameter), use the original name
            final_var_name = actual_var_name if actual_var_name else var_name

            if final_var_name not in line_numbers:
                line_numbers[final_var_name] = helper_line

        # Use the actual line from inside the helper, not the call line
        for pattern, helper_line in dynamic_patterns:
            line_number_regexes.append((pattern, helper_line))

    def _resolve_param_to_arg(
        self, func_def: ast.FunctionDef, call_node: ast.Call, param_name: str
    ) -> str | None:
        """Resolve a parameter name to the actual argument value from the call site.

        Args:
            func_def: The function definition
            call_node: The call to the function
            param_name: The parameter name to resolve

        Returns:
            The string value of the argument if it's a constant string, None otherwise
        """
        # Find the parameter index
        param_names = [arg.arg for arg in func_def.args.args]
        if param_name not in param_names:
            return None

        param_index = param_names.index(param_name)

        # Get the corresponding argument from the call
        if param_index < len(call_node.args):
            arg = call_node.args[param_index]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value

        # Check keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == param_name:
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    return keyword.value.value

        return None

    def _extract_from_imported_standalone_helper(
        self,
        module_name: str,
        func_name: str,
        call_node: ast.Call,
        call_line: int,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> list[tuple[str, int]]:
        """Extract from imported helper function called outside a loop.

        Returns tuples with the source filename for cross-file patterns.
        Also returns list of (var_name, call_line) for setattr-with-param patterns.
        """
        static_vars_from_call: list[tuple[str, int]] = []

        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                for node in ast.walk(module_tree):
                    if not (
                        isinstance(node, ast.FunctionDef) and node.name == func_name
                    ):
                        continue

                    # Extract patterns from the helper
                    ds_idx = self.function_analyzer.find_dataset_param_index(
                        node, call_node
                    )
                    if ds_idx is None or ds_idx >= len(node.args.args):
                        return static_vars_from_call

                    dataset_param_name = node.args.args[ds_idx].arg

                    # Check for setattr(dataset, param_name, ...) pattern
                    var_name_param_idx = (
                        self.operation_finder.find_setattr_with_param_index(
                            node, dataset_param_name
                        )
                    )

                    if var_name_param_idx is not None:
                        # Extract variable name from the call site
                        if var_name_param_idx < len(call_node.args):
                            arg = call_node.args[var_name_param_idx]
                            if isinstance(arg, ast.Constant) and isinstance(
                                arg.value, str
                            ):
                                # Return call site line for setattr-with-param pattern
                                static_vars_from_call.append((arg.value, call_line))
                    else:
                        # Original pattern extraction for non-setattr-param cases
                        _, dynamic_patterns = (
                            self.function_analyzer.extract_from_function(
                                node, dataset_param_name
                            )
                        )

                        # Return cross-file tuples with the helper file and line
                        rel_path = self.module_resolver.get_relative_path(module_file)
                        for pattern, helper_line in dynamic_patterns:
                            # Check if this is a template format pattern
                            if pattern == "__TEMPLATE_FORMAT__":
                                # Try to extract the template string from call arguments
                                resolved_pattern = self._resolve_template_from_call(
                                    node, call_node
                                )
                                if resolved_pattern:
                                    line_number_regexes.append(
                                        (resolved_pattern, (rel_path, helper_line))
                                    )
                            else:
                                line_number_regexes.append(
                                    (pattern, (rel_path, helper_line))
                                )

                    return static_vars_from_call
            except Exception:
                continue

        return static_vars_from_call

    def _extract_from_class_method(
        self,
        module_name: str,
        class_name: str,
        method_name: str,
        call_node: ast.Call,
        constructor_call: ast.Call | None,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract variables from a class method that modifies the dataset.

        Args:
            module_name: Name of the module containing the class
            class_name: Name of the class
            method_name: Name of the method being called
            call_node: The call node
            line_numbers: Dict to update with static variable line numbers
            line_number_regexes: List to update with dynamic patterns
        """
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                # Find the class definition
                class_def: ast.ClassDef | None = None
                for node in ast.walk(module_tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        class_def = node
                        break

                if not class_def:
                    return

                # Find the method in the class
                method_def: ast.FunctionDef | None = None
                for item in class_def.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        method_def = item
                        break

                if not method_def:
                    return

                # Extract variables from this method and any methods it calls
                self._extract_from_class_method_recursive(
                    class_def,
                    method_def,
                    module_file,
                    call_node,
                    constructor_call,
                    line_numbers,
                    line_number_regexes,
                )

                return
            except Exception:
                continue

    def _extract_from_class_method_recursive(
        self,
        class_def: ast.ClassDef,
        method_def: ast.FunctionDef,
        module_file: pathlib.Path,
        call_node: ast.Call,
        constructor_call: ast.Call | None,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
        visited_methods: set[str] | None = None,
    ) -> None:
        """Recursively extract variables from a class method and methods it calls.

        This handles patterns like:
        - method calls _helper(self.dataset, variable_name, ...)
        - _helper uses setattr(dataset, variable_name, ...)
        - method calls another_method which calls _helper
        """
        if visited_methods is None:
            visited_methods = set()

        # Avoid infinite recursion
        if method_def.name in visited_methods:
            return
        visited_methods.add(method_def.name)

        rel_path = self.module_resolver.get_relative_path(module_file)

        # Look for calls to other methods within this method
        for node in ast.walk(method_def):
            if not isinstance(node, ast.Call):
                continue

            # Check if this is a call to another method: self.method_name(...)
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "self":
                    helper_method_name = node.func.attr

                    # Find this helper method in the class
                    for item in class_def.body:
                        if (
                            isinstance(item, ast.FunctionDef)
                            and item.name == helper_method_name
                        ):
                            # Extract variables from the helper method
                            # Check if it uses setattr
                            self._extract_setattr_from_method(
                                class_def,
                                item,
                                node,
                                call_node,
                                constructor_call,
                                rel_path,
                                line_numbers,
                                line_number_regexes,
                            )

                            # Recursively process this method too
                            self._extract_from_class_method_recursive(
                                class_def,
                                item,
                                module_file,
                                call_node,
                                constructor_call,
                                line_numbers,
                                line_number_regexes,
                                visited_methods,
                            )

    def _extract_setattr_from_method(
        self,
        class_def: ast.ClassDef,
        method_def: ast.FunctionDef,
        method_call_node: ast.Call,
        original_call_node: ast.Call,
        constructor_call: ast.Call | None,
        rel_path: str,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract variables from a method that uses setattr.

        Args:
            class_def: The class definition containing the method
            method_def: The helper method definition (e.g., _update_dataset)
            method_call_node: The call to this method (e.g., self._update_dataset(...))
            original_call_node: The original method call on the instance
            constructor_call: The constructor call node (e.g., ClassName(...))
            rel_path: Relative path to the source file
            line_numbers: Dict to update
            line_number_regexes: List to update
        """
        # Look for setattr calls in the method
        for node in ast.walk(method_def):
            if not isinstance(node, ast.Call):
                continue

            if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
                continue

            if len(node.args) < 3:
                continue

            # The third argument is the variable name
            var_name_arg = node.args[1]

            # Check if it's a simple parameter reference
            if isinstance(var_name_arg, ast.Name):
                # Find which parameter this is
                param_name = var_name_arg.id
                param_index: int | None = None
                for idx, param in enumerate(method_def.args.args):
                    if param.arg == param_name:
                        param_index = idx
                        break

                if param_index is not None:
                    # Adjust for self parameter: method definitions have self as args[0],
                    # but method calls don't include self in the arguments
                    # So parameter index 0 (self) maps to no argument,
                    # parameter index 1 maps to method_call_node.args[0], etc.
                    call_arg_index = param_index - 1

                    if call_arg_index >= 0 and call_arg_index < len(
                        method_call_node.args
                    ):
                        # Get the actual argument from the method call
                        actual_arg = method_call_node.args[call_arg_index]

                        # Use the line number of the method call, not the setattr line
                        # This is clearer because it shows where the developer called
                        # the helper method, not the generic setattr implementation
                        call_line = method_call_node.lineno

                        # Check if it's a constant string
                        if isinstance(actual_arg, ast.Constant) and isinstance(
                            actual_arg.value, str
                        ):
                            # Static variable name
                            line_numbers[actual_arg.value] = (rel_path, call_line)

                        # Check if it's an instance attribute like self.codelist_name_1
                        elif isinstance(actual_arg, ast.Attribute) and isinstance(
                            actual_arg.value, ast.Name
                        ):
                            if actual_arg.value.id == "self" and constructor_call:
                                # Resolve the attribute from the constructor call
                                attr_name = actual_arg.attr
                                resolved_values = (
                                    self._resolve_instance_attribute_from_constructor(
                                        class_def, attr_name, constructor_call
                                    )
                                )
                                for value, value_line in resolved_values:
                                    # The constructor call is from the main file,
                                    # so use plain integer line numbers
                                    line_numbers[value] = value_line

                        # Check if it's an f-string
                        elif isinstance(actual_arg, ast.JoinedStr):
                            pattern = self.name_extractor.extract_from_fstring(
                                actual_arg
                            )
                            # Need to resolve any self.attribute references
                            resolved_pattern = (
                                self._resolve_instance_attributes_in_pattern(
                                    pattern, actual_arg, original_call_node
                                )
                            )
                            line_number_regexes.append(
                                (resolved_pattern, (rel_path, call_line))
                            )

    def _resolve_instance_attributes_in_pattern(
        self,
        pattern: str,
        fstring_node: ast.JoinedStr,
        call_node: ast.Call,
    ) -> str:
        """Resolve instance attributes in an f-string pattern.

        For example, f"{self.codelist_name_1}_{month}" where self.codelist_name_1
        is set from constructor arguments.
        """
        # For now, return the pattern as-is
        # A full implementation would track instance attributes through __init__
        # and resolve them from the constructor call arguments
        return pattern

    def _resolve_instance_attribute_from_constructor(
        self,
        class_def: ast.ClassDef,
        attr_name: str,
        constructor_call: ast.Call,
    ) -> list[tuple[str, int]]:
        """Resolve an instance attribute value from the constructor call.

        For example, if __init__ has:
            self.codelist_name_1, self.codelist_name_2 = codelist_names

        And the constructor is called with:
            ClassName(dataset, ("aspirin", "antiplatelet"))

        Then resolving "codelist_name_1" should return [("aspirin", 9)]
        where 9 is the line number of "aspirin" in the constructor call.

        Args:
            class_def: The class definition
            attr_name: The attribute name to resolve (e.g., "codelist_name_1")
            constructor_call: The constructor call node

        Returns:
            List of (value, line_number) tuples
        """
        # Find the __init__ method
        init_method: ast.FunctionDef | None = None
        for item in class_def.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break

        if not init_method:
            return []

        # Look for assignments to self.attr_name in __init__
        # Pattern 1: self.attr_name = something
        # Pattern 2: self.attr_name, self.other = tuple_param (unpacking)

        for node in ast.walk(init_method):
            if not isinstance(node, ast.Assign):
                continue

            # Check for tuple unpacking: self.a, self.b = param
            if isinstance(node.targets[0], ast.Tuple):
                # Find the index of our attribute in the tuple
                attr_index: int | None = None
                for idx, target_elem in enumerate(node.targets[0].elts):
                    if (
                        isinstance(target_elem, ast.Attribute)
                        and isinstance(target_elem.value, ast.Name)
                        and target_elem.value.id == "self"
                        and target_elem.attr == attr_name
                    ):
                        attr_index = idx
                        break

                if attr_index is not None:
                    # The RHS should be a parameter name
                    if isinstance(node.value, ast.Name):
                        param_name = node.value.id

                        # Find which parameter this is in __init__
                        param_index: int | None = None
                        for idx, param in enumerate(init_method.args.args):
                            if param.arg == param_name:
                                param_index = idx
                                break

                        if param_index is not None:
                            # Adjust for self: param_index 0 is self, 1 is first arg, etc.
                            call_arg_index = param_index - 1

                            if call_arg_index >= 0 and call_arg_index < len(
                                constructor_call.args
                            ):
                                # Get the argument from the constructor call
                                constructor_arg = constructor_call.args[call_arg_index]

                                # Check if it's a tuple
                                if isinstance(constructor_arg, ast.Tuple):
                                    if attr_index < len(constructor_arg.elts):
                                        tuple_elem = constructor_arg.elts[attr_index]

                                        # Extract the value if it's a constant string
                                        if isinstance(
                                            tuple_elem, ast.Constant
                                        ) and isinstance(tuple_elem.value, str):
                                            return [
                                                (tuple_elem.value, tuple_elem.lineno)
                                            ]

            # Check for simple assignment: self.attr_name = param
            elif isinstance(node.targets[0], ast.Attribute):
                target = node.targets[0]
                if (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == attr_name
                ):
                    # The RHS should be a parameter or value
                    if isinstance(node.value, ast.Name):
                        param_name = node.value.id

                        # Find which parameter this is in __init__
                        param_index = None
                        for idx, param in enumerate(init_method.args.args):
                            if param.arg == param_name:
                                param_index = idx
                                break

                        if param_index is not None:
                            call_arg_index = param_index - 1

                            if call_arg_index >= 0 and call_arg_index < len(
                                constructor_call.args
                            ):
                                constructor_arg = constructor_call.args[call_arg_index]

                                if isinstance(
                                    constructor_arg, ast.Constant
                                ) and isinstance(constructor_arg.value, str):
                                    return [
                                        (constructor_arg.value, constructor_arg.lineno)
                                    ]

        return []

    def _resolve_template_from_call(
        self, func_def: ast.FunctionDef, call_node: ast.Call
    ) -> str | None:
        """Resolve a template string pattern from the call site.

        Looks for a string argument passed to the function that contains format
        placeholders like {n}, {i}, etc., and converts it to a regex pattern.

        Args:
            func_def: Function definition
            call_node: Call node at the call site

        Returns:
            Regex pattern string, or None if not found
        """
        # Find which parameter receives the template string
        # Look for .format() calls in the function to identify the parameter
        template_param_name: str | None = None

        for node in ast.walk(func_def):
            if not isinstance(node, ast.Assign):
                continue

            # Look for pattern: var = param.format(...)
            if not isinstance(node.value, ast.Call):
                continue

            if not isinstance(node.value.func, ast.Attribute):
                continue

            if node.value.func.attr != "format":
                continue

            if isinstance(node.value.func.value, ast.Name):
                # This is the parameter that holds the template
                candidate = node.value.func.value.id
                param_names = [arg.arg for arg in func_def.args.args]
                if candidate in param_names:
                    template_param_name = candidate
                    break

        if not template_param_name:
            return None

        # Find the index of this parameter
        param_index = None
        for idx, param in enumerate(func_def.args.args):
            if param.arg == template_param_name:
                param_index = idx
                break

        if param_index is None or param_index >= len(call_node.args):
            return None

        # Extract the template string from the call arguments
        template_arg = call_node.args[param_index]

        if not (
            isinstance(template_arg, ast.Constant)
            and isinstance(template_arg.value, str)
        ):
            return None

        template_string = template_arg.value

        # Convert template string with placeholders like {n} to regex
        # First, replace {anything} with a placeholder marker
        import re as re_module

        placeholder_marker = "___PLACEHOLDER___"
        temp = re_module.sub(r"\{[^}]+\}", placeholder_marker, template_string)

        # Escape the rest of the string for regex
        escaped = re_module.escape(temp)

        # Replace the placeholder marker with .*
        pattern = escaped.replace(placeholder_marker, ".*")

        return pattern

    def extract_codelist_calls(
        self,
    ) -> dict[str, list[tuple[str | None, ...]]]:
        """Extract codelist_from_csv calls for each variable using comprehensive AST tracing.

        Returns:
            Dict mapping variable_name -> list of parameter tuples.
            Each tuple contains (arg1, arg2, ..., kwarg1_name=kwarg1_val, kwarg2_name=kwarg2_val, ...)
            All variables are included, even those with no codelist calls (empty list).
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(self.file_path))
        except Exception:
            return {}

        # Collect imports
        import_collector = ImportCollector()
        import_collector.collect(tree)
        import_collector.resolve_star_imports(self.module_resolver)

        # Result dictionary - will contain all variables
        variable_codelists: dict[str, list[tuple[str | None, ...]]] = {}

        # Find all dataset variable definitions and trace their expressions
        # Pass 1: Direct attribute assignments (dataset.var = expr)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # dataset.var_name = expression
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "dataset"
                        ):
                            var_name = target.attr
                            # Trace the expression comprehensively
                            codelist_calls = (
                                self.codelist_tracer.trace_expression_for_codelists(
                                    node.value, tree, import_collector, self.file_path
                                )
                            )
                            # Convert to tuples and deduplicate
                            codelists = [
                                self._codelist_call_to_tuple(call)
                                for call in codelist_calls
                            ]
                            # Deduplicate while preserving order
                            seen = set()
                            unique_codelists = []
                            for codelist in codelists:
                                if codelist not in seen:
                                    seen.add(codelist)
                                    unique_codelists.append(codelist)
                            variable_codelists[var_name] = unique_codelists

        # Pass 2: dataset.add_column() calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "dataset"
                        and node.func.attr == "add_column"
                    ):
                        if len(node.args) >= 2:
                            # First arg is variable name, second is expression
                            first_arg = node.args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(
                                first_arg.value, str
                            ):
                                var_name = first_arg.value
                                # Trace the expression comprehensively
                                codelist_calls = (
                                    self.codelist_tracer.trace_expression_for_codelists(
                                        node.args[1],
                                        tree,
                                        import_collector,
                                        self.file_path,
                                    )
                                )
                                # Convert to tuples and deduplicate
                                codelists = [
                                    self._codelist_call_to_tuple(call)
                                    for call in codelist_calls
                                ]
                                # Deduplicate while preserving order
                                seen = set()
                                unique_codelists = []
                                for codelist in codelists:
                                    if codelist not in seen:
                                        seen.add(codelist)
                                        unique_codelists.append(codelist)
                                variable_codelists[var_name] = unique_codelists

        # Pass 3: Check for variables defined in helper functions
        # This handles patterns like: dataset = create_dataset()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "dataset":
                        # Check if RHS is a function call
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                func_name = node.value.func.id
                                # Try to find variables defined in this function
                                if func_name in import_collector.function_defs:
                                    func_def = import_collector.function_defs[func_name]
                                    vars_from_func = (
                                        self._extract_codelists_from_function(
                                            func_def, tree, import_collector
                                        )
                                    )
                                    # Merge with existing
                                    self._merge_variable_codelists(
                                        variable_codelists, vars_from_func
                                    )

        # Pass 4: Helper functions that directly accept a dataset parameter
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            has_dataset_param = any(arg.arg == "dataset" for arg in node.args.args)
            if not has_dataset_param:
                has_dataset_param = any(
                    arg.arg == "dataset" for arg in node.args.kwonlyargs
                )
            if not has_dataset_param:
                continue

            vars_from_func = self._extract_codelists_from_function(
                node, tree, import_collector
            )
            self._merge_variable_codelists(variable_codelists, vars_from_func)

        # Pass 5: Module-level calls to helper functions that accept dataset
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Local helper call: helper(dataset, ...)
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in import_collector.function_defs:
                    func_def = import_collector.function_defs[func_name]
                    if self._call_uses_dataset_param(func_def, node):
                        call_values = self._function_call_string_values(func_def, node)
                        vars_from_func = self._extract_codelists_from_function(
                            func_def,
                            tree,
                            import_collector,
                            string_overrides=call_values,
                        )
                        self._merge_variable_codelists(
                            variable_codelists, vars_from_func
                        )

            # Imported module helper call: module.helper(dataset, ...)
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                obj_name = node.func.value.id
                func_name = node.func.attr
                if obj_name in import_collector.imported_modules:
                    module_name, original_name = import_collector.imported_modules[
                        obj_name
                    ]
                    # Only treat plain module imports (import variables)
                    if original_name is None:
                        vars_from_func = (
                            self._extract_codelists_from_imported_function_call(
                                module_name,
                                func_name,
                                node,
                            )
                        )
                        self._merge_variable_codelists(
                            variable_codelists, vars_from_func
                        )

        return variable_codelists

    def _resolve_string_literal_expr(
        self,
        expr: ast.AST,
        known_names: dict[str, str],
    ) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value

        if isinstance(expr, ast.Name):
            return known_names.get(expr.id)

        if isinstance(expr, ast.JoinedStr):
            parts: list[str] = []
            for value in expr.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    resolved = self._resolve_string_literal_expr(
                        value.value, known_names
                    )
                    if resolved is None:
                        return None
                    parts.append(resolved)
                else:
                    return None
            return "".join(parts)

        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = self._resolve_string_literal_expr(expr.left, known_names)
            right = self._resolve_string_literal_expr(expr.right, known_names)
            if left is None or right is None:
                return None
            return left + right

        return None

    def _function_default_string_values(
        self,
        func_def: ast.FunctionDef,
    ) -> dict[str, str]:
        default_values: dict[str, str] = {}

        positional_args = func_def.args.args
        positional_defaults = func_def.args.defaults
        if positional_defaults:
            start = len(positional_args) - len(positional_defaults)
            for arg, default in zip(positional_args[start:], positional_defaults):
                resolved = self._resolve_string_literal_expr(default, default_values)
                if resolved is not None:
                    default_values[arg.arg] = resolved

        for kwarg, default in zip(func_def.args.kwonlyargs, func_def.args.kw_defaults):
            if default is None:
                continue
            resolved = self._resolve_string_literal_expr(default, default_values)
            if resolved is not None:
                default_values[kwarg.arg] = resolved

        return default_values

    def _function_call_string_values(
        self,
        func_def: ast.FunctionDef,
        call_node: ast.Call,
    ) -> dict[str, str]:
        values: dict[str, str] = {}

        positional_params = [arg.arg for arg in func_def.args.args]
        for index, arg_node in enumerate(call_node.args):
            if index >= len(positional_params):
                break
            resolved = self._resolve_string_literal_expr(arg_node, values)
            if resolved is not None:
                values[positional_params[index]] = resolved

        for keyword in call_node.keywords:
            if not keyword.arg:
                continue
            resolved = self._resolve_string_literal_expr(keyword.value, values)
            if resolved is not None:
                values[keyword.arg] = resolved

        return values

    @staticmethod
    def _call_uses_dataset_param(
        func_def: ast.FunctionDef, call_node: ast.Call
    ) -> bool:
        positional_params = [arg.arg for arg in func_def.args.args]
        dataset_index = None
        for index, param in enumerate(positional_params):
            if param == "dataset":
                dataset_index = index
                break

        if dataset_index is not None and dataset_index < len(call_node.args):
            dataset_arg = call_node.args[dataset_index]
            if isinstance(dataset_arg, ast.Name):
                return True

        for keyword in call_node.keywords:
            if keyword.arg == "dataset" and isinstance(keyword.value, ast.Name):
                return True

        return False

    def _merge_variable_codelists(
        self,
        destination: dict[str, list[tuple[str | None, ...]]],
        source: dict[str, list[tuple[str | None, ...]]],
    ) -> None:
        for var_name, calls in source.items():
            if var_name in destination:
                existing = set(destination[var_name])
                for call in calls:
                    if call not in existing:
                        destination[var_name].append(call)
                        existing.add(call)
            else:
                destination[var_name] = calls

    def _extract_codelists_from_imported_function_call(
        self,
        module_name: str,
        func_name: str,
        call_node: ast.Call,
    ) -> dict[str, list[tuple[str | None, ...]]]:
        for module_file in self.module_resolver.find_module_file(module_name):
            if not module_file.exists():
                continue

            try:
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                module_import_collector = ImportCollector()
                module_import_collector.collect(module_tree)
                module_import_collector.resolve_star_imports(self.module_resolver)

                for node in ast.walk(module_tree):
                    if not (
                        isinstance(node, ast.FunctionDef) and node.name == func_name
                    ):
                        continue
                    if not self._call_uses_dataset_param(node, call_node):
                        return {}

                    call_string_values = self._function_call_string_values(
                        node, call_node
                    )
                    return self._extract_codelists_from_function(
                        node,
                        module_tree,
                        module_import_collector,
                        source_file=module_file,
                        string_overrides=call_string_values,
                    )
            except Exception:
                continue

        return {}

    def _extract_codelists_from_function(
        self,
        func_def: ast.FunctionDef,
        tree: ast.AST,
        import_collector: ImportCollector,
        source_file: pathlib.Path | None = None,
        string_overrides: dict[str, str] | None = None,
    ) -> dict[str, list[tuple[str | None, ...]]]:
        """Extract codelist calls from variables defined within a function.

        Args:
            func_def: Function definition node
            tree: Full AST tree
            import_collector: Import information

        Returns:
            Dict mapping variable names to codelist parameter tuples
        """
        variable_codelists: dict[str, list[tuple[str | None, ...]]] = {}
        default_string_values = self._function_default_string_values(func_def)
        if string_overrides:
            default_string_values.update(string_overrides)
        expression_file = source_file if source_file is not None else self.file_path

        # Look for dataset.var = expr patterns inside the function
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name):
                            # Could be dataset.var or any parameter name
                            var_name = target.attr
                            codelist_calls = (
                                self.codelist_tracer.trace_expression_for_codelists(
                                    node.value, tree, import_collector, expression_file
                                )
                            )
                            codelists = [
                                self._codelist_call_to_tuple(call)
                                for call in codelist_calls
                            ]
                            # Deduplicate while preserving order
                            seen = set()
                            unique_codelists = []
                            for codelist in codelists:
                                if codelist not in seen:
                                    seen.add(codelist)
                                    unique_codelists.append(codelist)
                            variable_codelists[var_name] = unique_codelists

            # Also check for add_column calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "add_column":
                        if len(node.args) >= 2:
                            first_arg = node.args[0]
                            var_name = self._resolve_string_literal_expr(
                                first_arg, default_string_values
                            )
                            if var_name is not None:
                                codelist_calls = (
                                    self.codelist_tracer.trace_expression_for_codelists(
                                        node.args[1],
                                        tree,
                                        import_collector,
                                        expression_file,
                                    )
                                )
                                codelists = [
                                    self._codelist_call_to_tuple(call)
                                    for call in codelist_calls
                                ]
                                # Deduplicate while preserving order
                                seen = set()
                                unique_codelists = []
                                for codelist in codelists:
                                    if codelist not in seen:
                                        seen.add(codelist)
                                        unique_codelists.append(codelist)
                                variable_codelists[var_name] = unique_codelists

        return variable_codelists

    def _extract_imported_codelists(
        self, import_collector: ImportCollector
    ) -> dict[str, list[CodelistCall]]:
        """Extract codelist definitions from imported modules.

        Args:
            import_collector: Import information

        Returns:
            Dict mapping codelist variable name -> list of CodelistCall objects
        """
        imported_codelists: dict[str, list[CodelistCall]] = {}

        # Check each imported module
        for imported_name, (
            module_name,
            original_name,
        ) in import_collector.imported_modules.items():
            # Find the module file
            for module_file in self.module_resolver.find_module_file(module_name):
                if not module_file.exists():
                    continue

                try:
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=str(module_file))

                    # Extract codelist calls from this module
                    module_codelists = CodelistCallFinder.extract_codelist_calls(
                        module_tree
                    )

                    # Map the imported names to local names
                    target_name = original_name or imported_name
                    if target_name in module_codelists:
                        imported_codelists[imported_name] = module_codelists[
                            target_name
                        ]

                    break
                except Exception:
                    continue

        # Handle star imports
        for star_module in import_collector.star_imports:
            for module_file in self.module_resolver.find_module_file(star_module):
                if not module_file.exists():
                    continue

                try:
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=str(module_file))

                    # Extract all codelist calls from this module
                    module_codelists = CodelistCallFinder.extract_codelist_calls(
                        module_tree
                    )

                    # Add all codelists from this module
                    for name, calls in module_codelists.items():
                        if name not in imported_codelists:
                            imported_codelists[name] = calls

                    break
                except Exception:
                    continue

        return imported_codelists

    def _find_codelists_in_expression(
        self,
        expr: ast.AST,
        codelist_assignments: dict[str, list[CodelistCall]],
    ) -> list[tuple[str | None, ...]]:
        """Find codelist references in an expression.

        Args:
            expr: Expression AST node
            codelist_assignments: Mapping of codelist variable names to their calls

        Returns:
            List of parameter tuples for each codelist call found
        """
        codelists: list[tuple[str | None, ...]] = []

        # Walk the expression tree
        for node in ast.walk(expr):
            # Look for Name nodes that reference codelist variables
            if isinstance(node, ast.Name):
                if node.id in codelist_assignments:
                    # Found a reference to a codelist variable
                    for codelist_call in codelist_assignments[node.id]:
                        # Convert CodelistCall to a flat tuple
                        param_tuple = self._codelist_call_to_tuple(codelist_call)
                        codelists.append(param_tuple)

        return codelists

    def _codelist_call_to_tuple(self, call: CodelistCall) -> tuple[str | None, ...]:
        """Convert a CodelistCall to a flat tuple of parameters.

        Args:
            call: CodelistCall object

        Returns:
            Tuple containing all positional args followed by formatted kwargs
        """
        result: list[str | None] = list(call.args)

        # Add keyword arguments as "key=value" strings
        for key, value in sorted(call.kwargs.items()):
            if value is not None:
                result.append(f"{key}={value}")
            else:
                result.append(f"{key}=<dynamic>")

        return tuple(result)

    def _ast_uses_name(self, node: ast.AST, name: str) -> bool:
        """Check if an AST node uses a specific variable name."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == name:
                return True
        return False

    def extract_codelist_calls_alt(self) -> dict[str, list[tuple[str | None, ...]]]:
        """Alternate extractor for codelist_from_csv calls.

        Uses the same high-level discovery flow as extract():
        - Find dataset creator functions and scan them
        - Scan module-level dataset operations
        - Scan standalone helpers called at module level that take the dataset

        For every dataset operation found, trace the defining expression using
        CodelistTracer to collect codelist_from_csv calls.

        Returns:
            Dict mapping variable_name -> list of codelist parameter tuples
        """
        results: dict[str, list[tuple[str | None, ...]]] = {}

        def add_calls(var_name: str, calls: list[CodelistCall]) -> None:
            if not calls:
                return
            if var_name not in results:
                results[var_name] = []
            for c in calls:
                results[var_name].append(self._codelist_call_to_tuple(c))

        try:
            with open(self.file_path, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(self.file_path))
        except Exception:
            return {}

        import_collector = ImportCollector()
        import_collector.collect(tree)
        import_collector.resolve_star_imports(self.module_resolver)

        tracer = self.codelist_tracer

        # Helpers to scan nodes for dataset operations and trace expressions
        def trace_from_assign_like(
            dataset_name: str,
            node: ast.Assign,
            owning_tree: ast.AST,
            owning_file: pathlib.Path,
        ) -> None:
            for target in node.targets:
                # dataset.attr = expr
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == dataset_name
                ):
                    var_name = target.attr
                    calls = tracer.trace_expression_for_codelists(
                        node.value, owning_tree, import_collector, owning_file
                    )
                    add_calls(var_name, calls)
                # dataset["name"] = expr
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == dataset_name
                ):
                    key = getattr(target, "slice", None)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        var_name = key.value
                        calls = tracer.trace_expression_for_codelists(
                            node.value, owning_tree, import_collector, owning_file
                        )
                        add_calls(var_name, calls)

        def trace_from_call_like(
            dataset_name: str,
            node: ast.Call,
            owning_tree: ast.AST,
            owning_file: pathlib.Path,
        ) -> None:
            # dataset.add_column(name, value)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == dataset_name
                and node.func.attr == "add_column"
                and len(node.args) >= 2
            ):
                name_arg, value_arg = node.args[0], node.args[1]
                if isinstance(name_arg, ast.Constant) and isinstance(
                    name_arg.value, str
                ):
                    var_name = name_arg.value
                    calls = tracer.trace_expression_for_codelists(
                        value_arg, owning_tree, import_collector, owning_file
                    )
                    add_calls(var_name, calls)

            # setattr(dataset, name, value)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 3
            ):
                obj, name_arg, value_arg = node.args[0], node.args[1], node.args[2]
                if isinstance(obj, ast.Name) and obj.id == dataset_name:
                    if isinstance(name_arg, ast.Constant) and isinstance(
                        name_arg.value, str
                    ):
                        var_name = name_arg.value
                        calls = tracer.trace_expression_for_codelists(
                            value_arg, owning_tree, import_collector, owning_file
                        )
                        add_calls(var_name, calls)

        def scan_getattr_loop_pattern(
            scope_node: ast.AST,
            dataset_name: str,
            lookup_tree: ast.AST,
        ) -> None:
            """Scan a scope for loops with getattr codelist pattern.

            Pattern:
                for disease in diseases:
                    disease_codelist = getattr(codelists, f"{disease}_snomed")
                    dataset.add_column(f"{disease}_count", func(disease_codelist))

            Also handles nested loops with if/elif branches where different branches
            assign to the same variable name with different getattr patterns.
            """

            def find_getattr_in_scope(
                scope: list[ast.stmt], var_name: str, loop_var_name: str
            ) -> ast.JoinedStr | None:
                """Find getattr pattern assigning to var_name within a scope."""
                for stmt in scope:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        target = stmt.targets[0]
                        if isinstance(target, ast.Name) and target.id == var_name:
                            if isinstance(stmt.value, ast.Call):
                                call = stmt.value
                                if (
                                    isinstance(call.func, ast.Name)
                                    and call.func.id == "getattr"
                                    and len(call.args) >= 2
                                ):
                                    second_arg = call.args[1]
                                    if isinstance(second_arg, ast.JoinedStr):
                                        uses_loop_var = any(
                                            isinstance(fval, ast.FormattedValue)
                                            and isinstance(fval.value, ast.Name)
                                            and fval.value.id == loop_var_name
                                            for fval in second_arg.values
                                        )
                                        if uses_loop_var:
                                            return second_arg
                    # Recurse into nested If statements
                    if isinstance(stmt, ast.If):
                        result = find_getattr_in_scope(
                            stmt.body, var_name, loop_var_name
                        )
                        if result:
                            return result
                        result = find_getattr_in_scope(
                            stmt.orelse, var_name, loop_var_name
                        )
                        if result:
                            return result
                    # Recurse into nested For loops
                    if isinstance(stmt, ast.For):
                        result = find_getattr_in_scope(
                            stmt.body, var_name, loop_var_name
                        )
                        if result:
                            return result
                return None

            def find_containing_branch(
                stmts: list[ast.stmt], target_node: ast.AST
            ) -> list[ast.stmt] | None:
                """Find the branch (list of statements) that contains target_node."""
                for stmt in stmts:
                    # Check if this statement IS the target
                    if stmt is target_node:
                        return stmts
                    # Check children
                    for child in ast.walk(stmt):
                        if child is target_node:
                            # Target is somewhere in this statement
                            # If it's an If, find which branch
                            if isinstance(stmt, ast.If):
                                result = find_containing_branch(stmt.body, target_node)
                                if result is not None:
                                    return result
                                result = find_containing_branch(
                                    stmt.orelse, target_node
                                )
                                if result is not None:
                                    return result
                            elif isinstance(stmt, ast.For):
                                result = find_containing_branch(stmt.body, target_node)
                                if result is not None:
                                    return result
                            # Found in this stmt but not a branch-able node
                            return stmts
                return None

            # Find For loops in the scope
            for_loops = []
            if isinstance(scope_node, ast.Module):
                for_loops = [
                    n
                    for n in ast.iter_child_nodes(scope_node)
                    if isinstance(n, ast.For)
                ]
            else:
                # For function bodies, look at direct children first, then walk
                for_loops = [n for n in ast.walk(scope_node) if isinstance(n, ast.For)]

            for loop_node in for_loops:
                # Check for simple Name iteration (for x in list_name)
                if not isinstance(loop_node.iter, ast.Name):
                    continue
                list_name = loop_node.iter.id

                # Get loop variable
                if not isinstance(loop_node.target, ast.Name):
                    continue
                loop_var_name = loop_node.target.id

                # Find the list definition - check both lookup_tree (main file) and scope
                list_values = self._find_list_literal_values(lookup_tree, list_name)
                if not list_values:
                    list_values = self._find_list_literal_values(scope_node, list_name)
                if not list_values:
                    continue

                # For each value in the list, simulate the loop
                for list_val in list_values:
                    # Find all add_column calls in the loop
                    for loop_child in ast.walk(loop_node):
                        if not isinstance(loop_child, ast.Call):
                            continue

                        if not (
                            isinstance(loop_child.func, ast.Attribute)
                            and isinstance(loop_child.func.value, ast.Name)
                            and loop_child.func.value.id == dataset_name
                            and loop_child.func.attr == "add_column"
                            and len(loop_child.args) >= 2
                        ):
                            continue

                        name_arg = loop_child.args[0]
                        value_arg = loop_child.args[1]

                        if not isinstance(name_arg, ast.JoinedStr):
                            continue

                        # Resolve variable name
                        var_name_parts = []
                        resolvable = True
                        for fval in name_arg.values:
                            if isinstance(fval, ast.Constant):
                                var_name_parts.append(str(fval.value))
                            elif isinstance(fval, ast.FormattedValue) and isinstance(
                                fval.value, ast.Name
                            ):
                                if fval.value.id == loop_var_name:
                                    var_name_parts.append(list_val)
                                else:
                                    resolvable = False
                                    break
                            else:
                                resolvable = False
                                break

                        if not resolvable:
                            continue

                        resolved_var_name = "".join(var_name_parts)

                        # Find which variables are used in value_arg
                        used_vars = set()
                        for node in ast.walk(value_arg):
                            if isinstance(node, ast.Name):
                                used_vars.add(node.id)

                        # For each used variable, try to find a getattr pattern in the same branch
                        for used_var in used_vars:
                            # Find the branch containing this add_column call
                            branch = find_containing_branch(loop_node.body, loop_child)
                            if branch is None:
                                branch = loop_node.body

                            # Look for getattr assignment to used_var in this branch
                            fstring_pattern = find_getattr_in_scope(
                                branch, used_var, loop_var_name
                            )
                            if fstring_pattern is None:
                                continue

                            # Resolve the codelist attribute name
                            attr_name_parts = []
                            for fval in fstring_pattern.values:
                                if isinstance(fval, ast.Constant):
                                    attr_name_parts.append(str(fval.value))
                                elif isinstance(
                                    fval, ast.FormattedValue
                                ) and isinstance(fval.value, ast.Name):
                                    if fval.value.id == loop_var_name:
                                        attr_name_parts.append(list_val)
                            resolved_attr_name = "".join(attr_name_parts)

                            # Get the module from the getattr call by searching in the branch
                            getattr_call = None
                            for stmt in branch:
                                if (
                                    isinstance(stmt, ast.Assign)
                                    and len(stmt.targets) == 1
                                    and isinstance(stmt.targets[0], ast.Name)
                                    and stmt.targets[0].id == used_var
                                    and isinstance(stmt.value, ast.Call)
                                    and isinstance(stmt.value.func, ast.Name)
                                    and stmt.value.func.id == "getattr"
                                ):
                                    getattr_call = stmt.value
                                    break
                                # Also check inside If statements in the branch
                                if isinstance(stmt, ast.If):
                                    for if_stmt in stmt.body + stmt.orelse:
                                        if (
                                            isinstance(if_stmt, ast.Assign)
                                            and len(if_stmt.targets) == 1
                                            and isinstance(if_stmt.targets[0], ast.Name)
                                            and if_stmt.targets[0].id == used_var
                                            and isinstance(if_stmt.value, ast.Call)
                                            and isinstance(if_stmt.value.func, ast.Name)
                                            and if_stmt.value.func.id == "getattr"
                                        ):
                                            getattr_call = if_stmt.value
                                            break
                                    if getattr_call:
                                        break

                            if getattr_call and len(getattr_call.args) >= 1:
                                module_node = getattr_call.args[0]
                                if isinstance(module_node, ast.Name):
                                    module_name = module_node.id
                                    if module_name in import_collector.imported_modules:
                                        actual_module, _ = (
                                            import_collector.imported_modules[
                                                module_name
                                            ]
                                        )
                                        calls = tracer._trace_imported_name(
                                            actual_module,
                                            resolved_attr_name,
                                            import_collector,
                                            0,
                                        )
                                        add_calls(resolved_var_name, calls)
                                        break

        def scan_function_body(
            func_def: ast.FunctionDef, dataset_name: str, owning_file: pathlib.Path
        ) -> None:
            for n in ast.walk(func_def):
                if isinstance(n, ast.Assign):
                    trace_from_assign_like(dataset_name, n, tree, owning_file)
                elif isinstance(n, ast.Call):
                    trace_from_call_like(dataset_name, n, tree, owning_file)
            # Also scan for getattr loop patterns inside the function
            scan_getattr_loop_pattern(func_def, dataset_name, tree)

        # Pass 1: dataset = creator_function(...); scan that function
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == "dataset"):
                    continue
                if not isinstance(node.value, ast.Call) or not isinstance(
                    node.value.func, ast.Name
                ):
                    continue
                func_name = node.value.func.id
                if func_name == "create_dataset":
                    continue
                # Local function
                if func_name in import_collector.function_defs:
                    scan_function_body(
                        import_collector.function_defs[func_name],
                        "dataset",
                        self.file_path,
                    )
                # Imported function
                elif func_name in import_collector.imported_modules:
                    module_name, original = import_collector.imported_modules[func_name]
                    target_name = original or func_name
                    for module_file in self.module_resolver.find_module_file(
                        module_name
                    ):
                        if not module_file.exists():
                            continue
                        try:
                            module_tree = ast.parse(
                                module_file.read_text(encoding="utf-8"),
                                filename=str(module_file),
                            )
                        except Exception:
                            continue
                        for fn in ast.walk(module_tree):
                            if (
                                isinstance(fn, ast.FunctionDef)
                                and fn.name == target_name
                            ):
                                # Inside a creator-like helper we assume local name is 'dataset'
                                for n in ast.walk(fn):
                                    if isinstance(n, ast.Assign):
                                        trace_from_assign_like(
                                            "dataset", n, module_tree, module_file
                                        )
                                    elif isinstance(n, ast.Call):
                                        trace_from_call_like(
                                            "dataset", n, module_tree, module_file
                                        )
                                break

        # Pass 2: Module-level direct dataset operations
        for top in ast.iter_child_nodes(tree):
            if isinstance(top, ast.FunctionDef):
                continue
            for n in ast.walk(top):
                if isinstance(n, ast.Assign):
                    trace_from_assign_like("dataset", n, tree, self.file_path)
                elif isinstance(n, ast.Call):
                    trace_from_call_like("dataset", n, tree, self.file_path)

        # Pass 3: Standalone helper function calls at module level with dataset arg
        for n in ast.iter_child_nodes(tree):
            if isinstance(n, ast.Call):
                # helper(dataset, ...)
                if (
                    isinstance(n.func, ast.Name)
                    and n.func.id in import_collector.function_defs
                ):
                    func_def = import_collector.function_defs[n.func.id]
                    ds_idx = self.function_analyzer.find_dataset_param_index(
                        func_def, n
                    )
                    if ds_idx is not None and ds_idx < len(func_def.args.args):
                        ds_param = func_def.args.args[ds_idx].arg
                        scan_function_body(func_def, ds_param, self.file_path)
                # module.helper(dataset, ...)
                elif isinstance(n.func, ast.Attribute) and isinstance(
                    n.func.value, ast.Name
                ):
                    mod_alias = n.func.value.id
                    func_name = n.func.attr
                    if mod_alias in import_collector.imported_modules:
                        module_name, _ = import_collector.imported_modules[mod_alias]
                        for module_file in self.module_resolver.find_module_file(
                            module_name
                        ):
                            if not module_file.exists():
                                continue
                            try:
                                module_src = module_file.read_text(encoding="utf-8")
                                module_tree = ast.parse(
                                    module_src, filename=str(module_file)
                                )
                            except Exception:
                                continue
                            for fn in ast.walk(module_tree):
                                if (
                                    isinstance(fn, ast.FunctionDef)
                                    and fn.name == func_name
                                ):
                                    ds_idx = (
                                        self.function_analyzer.find_dataset_param_index(
                                            fn, n
                                        )
                                    )
                                    if ds_idx is not None and ds_idx < len(
                                        fn.args.args
                                    ):
                                        ds_param = fn.args.args[ds_idx].arg
                                        # Use module_tree as owning tree and module_file as file path
                                        for sub in ast.walk(fn):
                                            if isinstance(sub, ast.Assign):
                                                trace_from_assign_like(
                                                    ds_param,
                                                    sub,
                                                    module_tree,
                                                    module_file,
                                                )
                                            elif isinstance(sub, ast.Call):
                                                trace_from_call_like(
                                                    ds_param,
                                                    sub,
                                                    module_tree,
                                                    module_file,
                                                )
                                    break

        # Pass 4: Loop-based dynamic variables with dict iteration
        # Pattern: for key, value in dict.items(): dataset.add_column(f"{key}_suffix", expr_using_value)
        # Also handles nested range() loops:
        #   for key, value in dict.items():
        #       for i in range(N):
        #           dataset.add_column(f"{key}_status{i}", expr)
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.For):
                continue

            # Check for dict.items() pattern
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"
            ):
                continue

            # Get the dict name
            if not isinstance(node.iter.func.value, ast.Name):
                continue
            dict_name = node.iter.func.value.id

            # Get loop variables (key, value)
            if not isinstance(node.target, ast.Tuple) or len(node.target.elts) < 2:
                continue
            key_var = node.target.elts[0]
            value_var = node.target.elts[1]
            if not isinstance(key_var, ast.Name) or not isinstance(value_var, ast.Name):
                continue
            key_var_name = key_var.id
            # value_var.id is the loop variable for dict values (not used in f-string resolution)

            # Find the dict definition
            dict_definition = self._find_dict_definition(tree, dict_name)
            if not dict_definition:
                continue

            # Find nested range() loops in the dict.items() loop body
            nested_range_loops: list[
                tuple[str, int]
            ] = []  # [(range_var_name, range_end), ...]
            for child in node.body:
                if isinstance(child, ast.For):
                    # Check for range(N) pattern
                    if (
                        isinstance(child.iter, ast.Call)
                        and isinstance(child.iter.func, ast.Name)
                        and child.iter.func.id == "range"
                        and len(child.iter.args) >= 1
                    ):
                        range_arg = child.iter.args[0]
                        if isinstance(range_arg, ast.Constant) and isinstance(
                            range_arg.value, int
                        ):
                            range_end = range_arg.value
                            if isinstance(child.target, ast.Name):
                                range_var_name = child.target.id
                                nested_range_loops.append((range_var_name, range_end))

            # For each (key, value) in the dict, simulate the loop iteration
            for dict_key_name, dict_value_node in dict_definition.items():
                # Look for dataset.add_column calls in the loop body
                for loop_child in ast.walk(node):
                    if not isinstance(loop_child, ast.Call):
                        continue

                    # Check for dataset.add_column(f"{key_var}_...", expr)
                    if not (
                        isinstance(loop_child.func, ast.Attribute)
                        and isinstance(loop_child.func.value, ast.Name)
                        and loop_child.func.value.id == "dataset"
                        and loop_child.func.attr == "add_column"
                        and len(loop_child.args) >= 2
                    ):
                        continue

                    name_arg = loop_child.args[0]
                    # value_arg (loop_child.args[1]) is traced via dict_value_node

                    # Check if name_arg is an f-string containing the key variable
                    if not isinstance(name_arg, ast.JoinedStr):
                        continue

                    # Check if this f-string contains both dict key and a range variable
                    fstring_vars: dict[str, str] = {}  # var_name -> placeholder
                    resolvable = True
                    uses_key_var = False

                    for fval in name_arg.values:
                        if isinstance(fval, ast.FormattedValue) and isinstance(
                            fval.value, ast.Name
                        ):
                            var_id = fval.value.id
                            if var_id == key_var_name:
                                uses_key_var = True
                                fstring_vars[var_id] = "KEY"
                            elif any(
                                var_id == rv_name for rv_name, _ in nested_range_loops
                            ):
                                fstring_vars[var_id] = "RANGE"
                            else:
                                # Unknown variable in f-string, can't resolve
                                resolvable = False
                                break

                    if not resolvable or not uses_key_var:
                        continue

                    # Check if the f-string uses a range variable
                    range_var_in_fstring = None
                    range_end_for_fstring = None
                    for var_id, placeholder in fstring_vars.items():
                        if placeholder == "RANGE":
                            range_var_in_fstring = var_id
                            # Find the corresponding range end
                            for rv_name, rv_end in nested_range_loops:
                                if rv_name == var_id:
                                    range_end_for_fstring = rv_end
                                    break
                            break

                    # Generate variable names
                    if range_var_in_fstring and range_end_for_fstring is not None:
                        # Nested range loop: generate variable for each range value
                        for range_val in range(range_end_for_fstring):
                            var_name_parts = []
                            for fval in name_arg.values:
                                if isinstance(fval, ast.Constant):
                                    var_name_parts.append(str(fval.value))
                                elif isinstance(
                                    fval, ast.FormattedValue
                                ) and isinstance(fval.value, ast.Name):
                                    var_id = fval.value.id
                                    if var_id == key_var_name:
                                        var_name_parts.append(dict_key_name)
                                    elif var_id == range_var_in_fstring:
                                        var_name_parts.append(str(range_val))

                            resolved_var_name = "".join(var_name_parts)

                            # Trace from the dict value node - this is a Name node
                            # referencing a variable (e.g., selected_medications_pre)
                            # that uses codelists in its definition
                            calls = tracer.trace_expression_for_codelists(
                                dict_value_node,
                                tree,
                                import_collector,
                                self.file_path,
                            )
                            add_calls(resolved_var_name, calls)
                    else:
                        # Simple case: just dict key variable, no nested range
                        var_name_parts = []
                        for fval in name_arg.values:
                            if isinstance(fval, ast.Constant):
                                var_name_parts.append(str(fval.value))
                            elif isinstance(fval, ast.FormattedValue) and isinstance(
                                fval.value, ast.Name
                            ):
                                if fval.value.id == key_var_name:
                                    var_name_parts.append(dict_key_name)
                                else:
                                    # Can't resolve this variable
                                    break
                        else:
                            resolved_var_name = "".join(var_name_parts)
                            calls = tracer.trace_expression_for_codelists(
                                dict_value_node,
                                tree,
                                import_collector,
                                self.file_path,
                            )
                            add_calls(resolved_var_name, calls)

        # Pass 5: Loop over list with getattr pattern for dynamic codelist access
        # Use the branch-aware scan_getattr_loop_pattern helper which handles
        # nested if/elif branches correctly
        scan_getattr_loop_pattern(tree, "dataset", tree)

        # Pass 6: Variables that reference other dataset variables via getattr(dataset, f"...")
        # Pattern:
        #   dataset.add_column(f"{disease}_inc_date",
        #       minimum_of(
        #           getattr(dataset, f"{disease}_prim_date", None),
        #           getattr(dataset, f"{disease}_sec_date", None)
        #       )
        #   )
        # The _inc_date variable should inherit codelists from _prim_date and _sec_date
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.For):
                continue

            # Check for simple Name iteration (for x in list_name)
            if not isinstance(node.iter, ast.Name):
                continue
            list_name = node.iter.id

            # Get loop variable
            if not isinstance(node.target, ast.Name):
                continue
            loop_var_name = node.target.id

            # Find the list definition
            list_values = self._find_list_literal_values(tree, list_name)
            if not list_values:
                continue

            # For each value in the list, find add_column calls that use getattr(dataset, ...)
            for list_val in list_values:
                for loop_child in ast.walk(node):
                    if not isinstance(loop_child, ast.Call):
                        continue

                    if not (
                        isinstance(loop_child.func, ast.Attribute)
                        and isinstance(loop_child.func.value, ast.Name)
                        and loop_child.func.value.id == "dataset"
                        and loop_child.func.attr == "add_column"
                        and len(loop_child.args) >= 2
                    ):
                        continue

                    name_arg = loop_child.args[0]
                    value_arg = loop_child.args[1]

                    # Resolve variable name
                    if not isinstance(name_arg, ast.JoinedStr):
                        continue

                    var_name_parts = []
                    resolvable = True
                    for fval in name_arg.values:
                        if isinstance(fval, ast.Constant):
                            var_name_parts.append(str(fval.value))
                        elif isinstance(fval, ast.FormattedValue) and isinstance(
                            fval.value, ast.Name
                        ):
                            if fval.value.id == loop_var_name:
                                var_name_parts.append(list_val)
                            else:
                                resolvable = False
                                break
                        else:
                            resolvable = False
                            break

                    if not resolvable:
                        continue

                    resolved_var_name = "".join(var_name_parts)

                    # Find all getattr(dataset, f"...") calls in value_arg
                    for call_node in ast.walk(value_arg):
                        if not isinstance(call_node, ast.Call):
                            continue
                        if not (
                            isinstance(call_node.func, ast.Name)
                            and call_node.func.id == "getattr"
                            and len(call_node.args) >= 2
                        ):
                            continue

                        # Check if first arg is 'dataset'
                        first_arg = call_node.args[0]
                        if not (
                            isinstance(first_arg, ast.Name)
                            and first_arg.id == "dataset"
                        ):
                            continue

                        # Resolve the second arg (the attribute name)
                        second_arg = call_node.args[1]
                        if isinstance(second_arg, ast.JoinedStr):
                            ref_name_parts = []
                            ref_resolvable = True
                            for fval in second_arg.values:
                                if isinstance(fval, ast.Constant):
                                    ref_name_parts.append(str(fval.value))
                                elif isinstance(
                                    fval, ast.FormattedValue
                                ) and isinstance(fval.value, ast.Name):
                                    if fval.value.id == loop_var_name:
                                        ref_name_parts.append(list_val)
                                    else:
                                        ref_resolvable = False
                                        break
                                else:
                                    ref_resolvable = False
                                    break

                            if ref_resolvable:
                                referenced_var = "".join(ref_name_parts)
                                # Inherit codelists from the referenced variable
                                if referenced_var in results:
                                    for codelist_call in results[referenced_var]:
                                        add_calls(
                                            resolved_var_name,
                                            [
                                                CodelistCall(
                                                    args=codelist_call[
                                                        : -len(
                                                            [
                                                                k
                                                                for k in codelist_call
                                                                if "=" in str(k)
                                                            ]
                                                        )
                                                    ]
                                                    if any(
                                                        "=" in str(k)
                                                        for k in codelist_call
                                                    )
                                                    else codelist_call,
                                                    kwargs={
                                                        k.split("=")[0]: k.split("=")[1]
                                                        for k in codelist_call
                                                        if isinstance(k, str)
                                                        and "=" in k
                                                    },
                                                )
                                            ],
                                        )

        # Pass 7: Loop with local dict intermediates for codelist storage
        # Pattern from disease_incidence/analysis/dataset_definition.py:
        #   for disease in diseases:
        #       snomed_inc_date = {}
        #       if hasattr(codelists, f"{disease}_snomed"):
        #           disease_codelist = getattr(codelists, f"{disease}_snomed")
        #           snomed_inc_date[f"{disease}_snomed_inc_date"] = first_code(disease_codelist).date
        #       dataset.add_column(f"{disease}_inc_date",
        #           minimum_of(snomed_inc_date[f"{disease}_snomed_inc_date"], ...))
        #
        # This pass:
        # 1. Finds dict subscript assignments: dict_var[f"key"] = expr
        # 2. Traces the expression for codelists (looking for getattr(codelists, ...) in scope)
        # 3. When add_column references dict_var[f"key"], inherits codelists from that entry
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.For):
                continue

            # Check for simple Name iteration (for x in list_name)
            if not isinstance(node.iter, ast.Name):
                continue
            list_name = node.iter.id

            # Get loop variable
            if not isinstance(node.target, ast.Name):
                continue
            loop_var_name = node.target.id

            # Find the list definition
            list_values = self._find_list_literal_values(tree, list_name)
            if not list_values:
                continue

            # For each list value, collect dict assignments and their codelists
            for list_val in list_values:
                # Track dict subscript assignments: {(dict_name, resolved_key): [CodelistCall, ...]}
                dict_subscript_codelists: dict[tuple[str, str], list[CodelistCall]] = {}

                # First pass: find all dict subscript assignments in the loop
                for loop_stmt in ast.walk(node):
                    if not isinstance(loop_stmt, ast.Assign):
                        continue
                    if len(loop_stmt.targets) != 1:
                        continue
                    target = loop_stmt.targets[0]

                    # Check for dict[f"key"] = expr pattern
                    if not isinstance(target, ast.Subscript):
                        continue
                    if not isinstance(target.value, ast.Name):
                        continue

                    dict_name = target.value.id
                    key_node = target.slice

                    # Resolve the key if it's an f-string
                    if isinstance(key_node, ast.JoinedStr):
                        key_parts = []
                        resolvable = True
                        for fval in key_node.values:
                            if isinstance(fval, ast.Constant):
                                key_parts.append(str(fval.value))
                            elif isinstance(fval, ast.FormattedValue) and isinstance(
                                fval.value, ast.Name
                            ):
                                if fval.value.id == loop_var_name:
                                    key_parts.append(list_val)
                                else:
                                    resolvable = False
                                    break
                            else:
                                resolvable = False
                                break

                        if not resolvable:
                            continue

                        resolved_key = "".join(key_parts)
                    elif isinstance(key_node, ast.Constant) and isinstance(
                        key_node.value, str
                    ):
                        resolved_key = key_node.value
                    else:
                        continue

                    # Look for getattr(codelists, ...) in the containing branch
                    # to find the codelist being used
                    def find_getattr_codelists_in_branch(
                        stmts: list[ast.stmt], target_assign: ast.Assign
                    ) -> list[CodelistCall]:
                        """Find getattr(codelists, ...) calls in the same branch as target_assign."""
                        calls_found: list[CodelistCall] = []

                        # Find the branch containing target_assign
                        def find_branch(
                            stmts_list: list[ast.stmt],
                        ) -> list[ast.stmt] | None:
                            for stmt in stmts_list:
                                if stmt is target_assign:
                                    return stmts_list
                                if isinstance(stmt, ast.If):
                                    result = find_branch(stmt.body)
                                    if result is not None:
                                        return result
                                    result = find_branch(stmt.orelse)
                                    if result is not None:
                                        return result
                                elif isinstance(stmt, ast.For):
                                    result = find_branch(stmt.body)
                                    if result is not None:
                                        return result
                            return None

                        branch = find_branch(stmts)
                        if branch is None:
                            return calls_found

                        # Look for getattr assignments in this branch
                        for stmt in branch:
                            if not isinstance(stmt, ast.Assign):
                                continue
                            if len(stmt.targets) != 1:
                                continue
                            if not isinstance(stmt.targets[0], ast.Name):
                                continue

                            # Check if this is a getattr(codelists, f"...") call
                            if isinstance(stmt.value, ast.Call):
                                call = stmt.value
                                if (
                                    isinstance(call.func, ast.Name)
                                    and call.func.id == "getattr"
                                    and len(call.args) >= 2
                                ):
                                    first_arg = call.args[0]
                                    second_arg = call.args[1]

                                    # Check if first arg is 'codelists' module
                                    if isinstance(first_arg, ast.Name):
                                        module_name = first_arg.id
                                        if (
                                            module_name
                                            in import_collector.imported_modules
                                        ):
                                            # Resolve the codelist attribute name
                                            if isinstance(second_arg, ast.JoinedStr):
                                                attr_parts = []
                                                for fval in second_arg.values:
                                                    if isinstance(fval, ast.Constant):
                                                        attr_parts.append(
                                                            str(fval.value)
                                                        )
                                                    elif isinstance(
                                                        fval, ast.FormattedValue
                                                    ) and isinstance(
                                                        fval.value, ast.Name
                                                    ):
                                                        if (
                                                            fval.value.id
                                                            == loop_var_name
                                                        ):
                                                            attr_parts.append(list_val)

                                                resolved_attr = "".join(attr_parts)
                                                actual_module, _ = (
                                                    import_collector.imported_modules[
                                                        module_name
                                                    ]
                                                )
                                                traced_calls = (
                                                    tracer._trace_imported_name(
                                                        actual_module,
                                                        resolved_attr,
                                                        import_collector,
                                                        0,
                                                    )
                                                )
                                                calls_found.extend(traced_calls)

                        return calls_found

                    # Find codelists in the containing branch
                    branch_codelists = find_getattr_codelists_in_branch(
                        node.body, loop_stmt
                    )
                    if branch_codelists:
                        dict_subscript_codelists[(dict_name, resolved_key)] = (
                            branch_codelists
                        )

                # Second pass: find add_column calls that reference dict subscripts
                for loop_child in ast.walk(node):
                    if not isinstance(loop_child, ast.Call):
                        continue

                    if not (
                        isinstance(loop_child.func, ast.Attribute)
                        and isinstance(loop_child.func.value, ast.Name)
                        and loop_child.func.value.id == "dataset"
                        and loop_child.func.attr == "add_column"
                        and len(loop_child.args) >= 2
                    ):
                        continue

                    name_arg = loop_child.args[0]
                    value_arg = loop_child.args[1]

                    # Resolve variable name
                    if not isinstance(name_arg, ast.JoinedStr):
                        continue

                    var_name_parts = []
                    resolvable = True
                    for fval in name_arg.values:
                        if isinstance(fval, ast.Constant):
                            var_name_parts.append(str(fval.value))
                        elif isinstance(fval, ast.FormattedValue) and isinstance(
                            fval.value, ast.Name
                        ):
                            if fval.value.id == loop_var_name:
                                var_name_parts.append(list_val)
                            else:
                                resolvable = False
                                break
                        else:
                            resolvable = False
                            break

                    if not resolvable:
                        continue

                    resolved_var_name = "".join(var_name_parts)

                    # Find dict subscript references in value_arg
                    for sub_node in ast.walk(value_arg):
                        if not isinstance(sub_node, ast.Subscript):
                            continue
                        if not isinstance(sub_node.value, ast.Name):
                            continue

                        ref_dict_name = sub_node.value.id
                        ref_key_node = sub_node.slice

                        # Resolve the referenced key
                        if isinstance(ref_key_node, ast.JoinedStr):
                            ref_key_parts = []
                            ref_resolvable = True
                            for fval in ref_key_node.values:
                                if isinstance(fval, ast.Constant):
                                    ref_key_parts.append(str(fval.value))
                                elif isinstance(
                                    fval, ast.FormattedValue
                                ) and isinstance(fval.value, ast.Name):
                                    if fval.value.id == loop_var_name:
                                        ref_key_parts.append(list_val)
                                    else:
                                        ref_resolvable = False
                                        break
                                else:
                                    ref_resolvable = False
                                    break

                            if not ref_resolvable:
                                continue

                            resolved_ref_key = "".join(ref_key_parts)
                        elif isinstance(ref_key_node, ast.Constant) and isinstance(
                            ref_key_node.value, str
                        ):
                            resolved_ref_key = ref_key_node.value
                        else:
                            continue

                        # Look up codelists for this dict reference
                        lookup_key = (ref_dict_name, resolved_ref_key)
                        if lookup_key in dict_subscript_codelists:
                            add_calls(
                                resolved_var_name,
                                dict_subscript_codelists[lookup_key],
                            )

        # Pass 8: Propagate codelists to variables that reference other dataset variables
        # via getattr(dataset, f"...") - must run after Pass 7 populates results
        # This is a second iteration of Pass 6 logic to handle cases where
        # the referenced variable was added by Pass 7
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.For):
                continue

            if not isinstance(node.iter, ast.Name):
                continue
            list_name = node.iter.id

            if not isinstance(node.target, ast.Name):
                continue
            loop_var_name = node.target.id

            list_values = self._find_list_literal_values(tree, list_name)
            if not list_values:
                continue

            for list_val in list_values:
                for loop_child in ast.walk(node):
                    if not isinstance(loop_child, ast.Call):
                        continue

                    if not (
                        isinstance(loop_child.func, ast.Attribute)
                        and isinstance(loop_child.func.value, ast.Name)
                        and loop_child.func.value.id == "dataset"
                        and loop_child.func.attr == "add_column"
                        and len(loop_child.args) >= 2
                    ):
                        continue

                    name_arg = loop_child.args[0]
                    value_arg = loop_child.args[1]

                    if not isinstance(name_arg, ast.JoinedStr):
                        continue

                    var_name_parts = []
                    resolvable = True
                    for fval in name_arg.values:
                        if isinstance(fval, ast.Constant):
                            var_name_parts.append(str(fval.value))
                        elif isinstance(fval, ast.FormattedValue) and isinstance(
                            fval.value, ast.Name
                        ):
                            if fval.value.id == loop_var_name:
                                var_name_parts.append(list_val)
                            else:
                                resolvable = False
                                break
                        else:
                            resolvable = False
                            break

                    if not resolvable:
                        continue

                    resolved_var_name = "".join(var_name_parts)

                    # Find all getattr(dataset, f"...") calls in value_arg
                    for call_node in ast.walk(value_arg):
                        if not isinstance(call_node, ast.Call):
                            continue
                        if not (
                            isinstance(call_node.func, ast.Name)
                            and call_node.func.id == "getattr"
                            and len(call_node.args) >= 2
                        ):
                            continue

                        first_arg = call_node.args[0]
                        if not (
                            isinstance(first_arg, ast.Name)
                            and first_arg.id == "dataset"
                        ):
                            continue

                        second_arg = call_node.args[1]
                        if isinstance(second_arg, ast.JoinedStr):
                            ref_name_parts = []
                            ref_resolvable = True
                            for fval in second_arg.values:
                                if isinstance(fval, ast.Constant):
                                    ref_name_parts.append(str(fval.value))
                                elif isinstance(
                                    fval, ast.FormattedValue
                                ) and isinstance(fval.value, ast.Name):
                                    if fval.value.id == loop_var_name:
                                        ref_name_parts.append(list_val)
                                    else:
                                        ref_resolvable = False
                                        break
                                else:
                                    ref_resolvable = False
                                    break

                            if ref_resolvable:
                                referenced_var = "".join(ref_name_parts)
                                # Inherit codelists from the referenced variable
                                if referenced_var in results:
                                    for codelist_call in results[referenced_var]:
                                        add_calls(
                                            resolved_var_name,
                                            [
                                                CodelistCall(
                                                    args=codelist_call[
                                                        : -len(
                                                            [
                                                                k
                                                                for k in codelist_call
                                                                if "=" in str(k)
                                                            ]
                                                        )
                                                    ]
                                                    if any(
                                                        "=" in str(k)
                                                        for k in codelist_call
                                                    )
                                                    else codelist_call,
                                                    kwargs={
                                                        k.split("=")[0]: k.split("=")[1]
                                                        for k in codelist_call
                                                        if isinstance(k, str)
                                                        and "=" in k
                                                    },
                                                )
                                            ],
                                        )

        return results


def extract_variable_line_numbers(
    file_path: pathlib.Path, repo_root: pathlib.Path
) -> tuple[dict[str, int | tuple[str, int]], list[tuple[str, int | tuple[str, int]]]]:
    """Extract variable definitions from an ehrQL dataset definition file.

    This is the main entry point that maintains backward compatibility with
    the original function signature.

    Args:
        file_path: Absolute path to the dataset definition file
        repo_root: Absolute path to the repository root

    Returns:
        Tuple of:
        - dict mapping variable_name -> line_number (int) or (filename, line_number)
        - list of (regex_pattern, line_number_or_tuple) for dynamic variables
          where line_number_or_tuple is int for same-file or (filename, line) for cross-file
    """
    extractor = VariableExtractor(file_path, repo_root)
    return extractor.extract()


def extract_variable_codelists(
    file_path: pathlib.Path, repo_root: pathlib.Path
) -> dict[str, list[tuple[str | None, ...]]]:
    """Extract codelist_from_csv calls for each variable in an ehrQL dataset file.

    Args:
        file_path: Absolute path to the dataset definition file
        repo_root: Absolute path to the repository root

    Returns:
        Dict mapping variable_name -> list of parameter tuples.
        Each tuple contains the parameters passed to codelist_from_csv:
        (filepath, "column=value", "system=value", etc.)
    """
    extractor = VariableExtractor(file_path, repo_root)
    return extractor.extract_codelist_calls()
