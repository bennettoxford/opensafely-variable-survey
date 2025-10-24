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
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Find setattr(dataset, name, value) calls.

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

        return static_vars, dynamic_patterns

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

            # Check for setattr calls
            static, dynamic = self.operation_finder.find_setattr_calls(
                node, dataset_param_name
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
        self.imported_modules: dict[str, tuple[str, str | None]] = {}
        self.star_imports: list[str] = []

    def collect(self, tree: ast.AST) -> None:
        """Walk AST and collect all imports and function definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.function_defs[node.name] = node

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
        """Resolve star imports by loading the modules and finding functions."""
        for star_module in self.star_imports:
            for module_file in module_resolver.find_module_file(star_module):
                if not module_file.exists():
                    continue

                try:
                    with open(module_file, encoding="utf-8") as f:
                        module_source = f.read()
                    module_tree = ast.parse(module_source, filename=str(module_file))

                    # Add all functions from this module
                    for node in ast.walk(module_tree):
                        if isinstance(node, ast.FunctionDef):
                            # Map function name to its module
                            self.imported_modules[node.name] = (star_module, node.name)
                    break  # Only process first found file
                except Exception:
                    continue


class VariableExtractor:
    """Main class for extracting variable definitions from ehrQL files."""

    def __init__(self, file_path: pathlib.Path, repo_root: pathlib.Path):
        self.file_path = file_path
        self.repo_root = repo_root
        self.module_resolver = ModuleResolver(file_path, repo_root)
        self.name_extractor = DynamicNameExtractor()
        self.operation_finder = DatasetOperationFinder(self.name_extractor)
        self.function_analyzer = FunctionAnalyzer(self.operation_finder)

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
        self._extract_from_loops(tree, import_collector, line_number_regexes)

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
        """Extract direct module-level dataset operations."""
        for node in ast.walk(tree):
            # Attribute assignments
            for var_name, line in self.operation_finder.find_variable_assignments(node):
                if var_name not in line_numbers:
                    line_numbers[var_name] = line

            # add_column calls
            static, dynamic = self.operation_finder.find_add_column_calls(node)
            for var_name, line in static:
                if var_name not in line_numbers:
                    line_numbers[var_name] = line
            for pattern, line in dynamic:
                line_number_regexes.append((pattern, line))

    def _extract_from_loops(
        self,
        tree: ast.AST,
        import_collector: ImportCollector,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract dynamic variables from loops."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.For, ast.While)):
                self._process_loop(node, import_collector, line_number_regexes)

    def _process_loop(
        self,
        loop_node: ast.For | ast.While,
        import_collector: ImportCollector,
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Process a single loop node to find dynamic variables."""
        loop_line = loop_node.lineno

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
                            func_def, child, call_line, line_number_regexes
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

                # Module.function() call
                elif isinstance(child.func, ast.Attribute) and isinstance(
                    child.func.value, ast.Name
                ):
                    mod_alias = child.func.value.id
                    func_name = child.func.attr

                    if mod_alias in import_collector.imported_modules:
                        module_name, _ = import_collector.imported_modules[mod_alias]
                        static_vars = self._extract_from_imported_standalone_helper(
                            module_name,
                            func_name,
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
        line_number_regexes: list[tuple[str, int | tuple[str, int]]],
    ) -> None:
        """Extract from helper function called outside a loop.

        Returns the line numbers from inside the helper function, not the call line.
        """
        ds_idx = self.function_analyzer.find_dataset_param_index(func_def, call_node)
        if ds_idx is None or ds_idx >= len(func_def.args.args):
            return

        dataset_param_name = func_def.args.args[ds_idx].arg
        _, dynamic_patterns = self.function_analyzer.extract_from_function(
            func_def, dataset_param_name
        )

        # Use the actual line from inside the helper, not the call line
        for pattern, helper_line in dynamic_patterns:
            line_number_regexes.append((pattern, helper_line))

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
                            line_number_regexes.append(
                                (pattern, (rel_path, helper_line))
                            )

                    return static_vars_from_call
            except Exception:
                continue

        return static_vars_from_call


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
