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


class CodelistTracer:
    """Traces codelist_from_csv calls through variable and function calls."""

    INLINE_CODELIST_SENTINEL = "<inline>"

    def __init__(self, module_resolver: ModuleResolver):
        self.module_resolver = module_resolver
        self._visited_vars: set[tuple[str, str]] = set()  # (file_path, var_name)
        self._codelist_cache: dict[tuple[str, str], list[CodelistCall]] = {}

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
        return self._trace_expression(expr, tree, import_collector, file_path)

    def _trace_expression(
        self,
        expr: ast.AST,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
        depth: int = 0,
    ) -> list[CodelistCall]:
        """Recursively trace an expression to find all codelist_from_csv calls.

        Args:
            expr: Expression to trace
            tree: AST tree of current file
            import_collector: Import information
            file_path: Current file path
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
                    node.id, tree, import_collector, file_path, depth
                )
                codelist_calls.extend(calls)

            # Attribute access - could be accessing codelist from class/module
            if isinstance(node, ast.Attribute):
                calls = self._trace_attribute(
                    node, tree, import_collector, file_path, depth
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
        depth: int,
    ) -> list[CodelistCall]:
        """Trace a Name reference to find codelists.

        Args:
            name: Variable name to trace
            tree: AST tree of current file
            import_collector: Import information
            file_path: Current file path
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
            # Look for local variable definition
            calls = self._find_local_definition(
                name, tree, import_collector, file_path, depth
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
                # Find the dataset variable assignment and trace its expression
                calls = self._find_dataset_variable_reference(
                    attr_name, tree, import_collector, file_path, depth
                )
                codelist_calls.extend(calls)
            else:
                # Check if it's a local class
                calls = self._trace_local_class_attribute(
                    obj_name, attr_name, tree, import_collector, file_path, depth
                )
                codelist_calls.extend(calls)

        # Handle nested attributes like obj.attr1.attr2
        elif isinstance(attr_node.value, ast.Attribute):
            calls = self._trace_attribute(
                attr_node.value, tree, import_collector, file_path, depth
            )
            codelist_calls.extend(calls)

        return codelist_calls

    def _find_dataset_variable_reference(
        self,
        var_name: str,
        tree: ast.AST,
        import_collector: ImportCollector,
        file_path: pathlib.Path,
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
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        # Prevent infinite recursion
        if depth > 50:
            return []

        codelist_calls: list[CodelistCall] = []

        # Look for dataset.var_name = expression patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if target is dataset.var_name
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "dataset"
                            and target.attr == var_name
                        ):
                            # Found the assignment, trace the right-hand side
                            calls = self._trace_expression(
                                node.value,
                                tree,
                                import_collector,
                                file_path,
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
        depth: int,
    ) -> list[CodelistCall]:
        """Find and trace a local variable definition.

        Args:
            var_name: Variable name to find
            tree: AST tree to search
            import_collector: Import information
            file_path: Current file path
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        codelist_calls: list[CodelistCall] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue

            # Check if this assigns to our variable
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    # Recursively trace the RHS
                    calls = self._trace_expression(
                        node.value, tree, import_collector, file_path, depth + 1
                    )
                    codelist_calls.extend(calls)

        return codelist_calls

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
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                # Create import collector for this module
                module_import_collector = ImportCollector()
                module_import_collector.collect(module_tree)
                module_import_collector.resolve_star_imports(self.module_resolver)

                # Find the definition in this module
                return self._find_local_definition(
                    target_name,
                    module_tree,
                    module_import_collector,
                    module_file,
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
                with open(module_file, encoding="utf-8") as f:
                    module_source = f.read()
                module_tree = ast.parse(module_source, filename=str(module_file))

                # Create import collector for this module
                module_import_collector = ImportCollector()
                module_import_collector.collect(module_tree)
                module_import_collector.resolve_star_imports(self.module_resolver)

                # Find the class definition
                for node in ast.walk(module_tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        # Look for the attribute in the class
                        return self._find_class_attribute_definition(
                            node,
                            attr_name,
                            module_tree,
                            module_import_collector,
                            module_file,
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
        depth: int,
    ) -> list[CodelistCall]:
        """Find an attribute definition within a class.

        Args:
            class_node: AST ClassDef node
            attr_name: Attribute name to find
            tree: Full AST tree
            import_collector: Import information
            file_path: Current file path
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
                            node.value, tree, import_collector, file_path, depth + 1
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
        depth: int,
    ) -> list[CodelistCall]:
        """Trace an attribute of a locally-defined class.

        Args:
            class_name: Name of the local class
            attr_name: Attribute name to find
            tree: AST tree
            import_collector: Import information
            file_path: Current file path
            depth: Recursion depth

        Returns:
            List of CodelistCall objects
        """
        # Find the class definition in the local tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return self._find_class_attribute_definition(
                    node, attr_name, tree, import_collector, file_path, depth
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
                    node, import_collector, line_numbers, line_number_regexes
                )

    def _process_loop(
        self,
        loop_node: ast.For | ast.While,
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Process a single loop node to find dynamic variables."""
        loop_line = loop_node.lineno

        # Check if this is a dict.items() loop pattern
        if self._extract_from_dict_items_loop(
            loop_node, import_collector, line_numbers
        ):
            # Successfully extracted from dict.items() pattern, nothing more to do
            return

        for child in ast.walk(loop_node):
            # Direct dataset operations in loop
            _, dynamic = self.operation_finder.find_add_column_calls(child)
            for pattern, _ in dynamic:
                line_number_regexes.append((pattern, loop_line))

            _, dynamic = self.operation_finder.find_setattr_calls(child)
            for pattern, _ in dynamic:
                line_number_regexes.append((pattern, loop_line))

            _, dynamic = self.operation_finder.find_subscript_assignments(child)
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
        import_collector: ImportCollector,
        line_numbers: dict[str, int | tuple[str, int]],
    ) -> bool:
        """Extract variables from a dict.items() loop pattern.

        Pattern:
            for var_name, var_value in imported_dict.items():
                setattr(dataset, var_name, var_value)

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

        # Check if the dict is imported
        if dict_name not in import_collector.imported_modules:
            return False

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

        if not setattr_node or len(setattr_node.args) < 3:
            return False

        # Now we need to find the dict definition in the imported module
        # and extract the variable names and their definition line numbers
        module_candidates = self.module_resolver.find_module_file(module_name)
        module_path = None
        for candidate in module_candidates:
            if candidate.exists():
                module_path = candidate
                break

        if not module_path:
            return False

        try:
            with open(module_path) as f:
                module_source = f.read()
            module_tree = ast.parse(module_source)
        except (OSError, SyntaxError):
            return False

        # Find the dict assignment in the module
        dict_definition = self._find_dict_definition(module_tree, target_dict_name)
        if not dict_definition:
            return False

        # Extract variable definitions from the dict
        rel_path = str(pathlib.Path(module_path).relative_to(self.repo_root))
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

    def _find_dict_definition(
        self, tree: ast.AST, dict_name: str
    ) -> dict[str, ast.AST] | None:
        """Find a dict definition and return its keys and values.

        Looks for patterns like:
            my_dict = dict(key1=value1, key2=value2)
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
                        self._extract_from_class_method(
                            module_name,
                            class_name,
                            method_or_func_name,
                            child,
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
                                item,
                                node,
                                call_node,
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
                                line_numbers,
                                line_number_regexes,
                                visited_methods,
                            )

    def _extract_setattr_from_method(
        self,
        method_def: ast.FunctionDef,
        method_call_node: ast.Call,
        original_call_node: ast.Call,
        rel_path: str,
        line_numbers: dict[str, int | tuple[str, int]],
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract variables from a method that uses setattr.

        Args:
            method_def: The helper method definition (e.g., _update_dataset)
            method_call_node: The call to this method (e.g., self._update_dataset(...))
            original_call_node: The original method call on the instance
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
                                    for var, calls in vars_from_func.items():
                                        if var not in variable_codelists:
                                            variable_codelists[var] = calls

        return variable_codelists

    def _extract_codelists_from_function(
        self,
        func_def: ast.FunctionDef,
        tree: ast.AST,
        import_collector: ImportCollector,
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
                                    node.value, tree, import_collector, self.file_path
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
                            if isinstance(first_arg, ast.Constant) and isinstance(
                                first_arg.value, str
                            ):
                                var_name = first_arg.value
                                codelist_calls = (
                                    self.codelist_tracer.trace_expression_for_codelists(
                                        node.args[1],
                                        tree,
                                        import_collector,
                                        self.file_path,
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
