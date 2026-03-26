# from __future__ import annotations

# from dataclasses import dataclass, asdict
# from typing import List, Dict, Any
# import ast

# from src.chunking.fixed_chunker import chunk_code_fixed


# @dataclass
# class StructureChunk:
#     chunk_id: str
#     repo: str
#     path: str
#     language: str
#     chunk_type: str
#     symbol_name: str
#     start_line: int
#     end_line: int
#     text: str

#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)


# # def _safe_get_source_segment(lines: List[str], start_line: int, end_line: int) -> str:
# #     start_idx = max(0, start_line - 1)
# #     end_idx = min(len(lines), end_line)
# #     return "\n".join(lines[start_idx:end_idx]).strip()

# def _safe_get_source_segment(lines: List[str], start_line: int, end_line: int) -> str:
#     start_idx = max(0, start_line - 1)
#     end_idx = min(len(lines), end_line)
#     return "\n".join(lines[start_idx:end_idx]).rstrip()


# def _get_end_lineno(node: ast.AST) -> int:
#     end_lineno = getattr(node, "end_lineno", None)
#     if end_lineno is None:
#         return getattr(node, "lineno", 1)
#     return end_lineno


# # def chunk_code_structure_aware(
# #     code: str,
# #     repo: str,
# #     path: str,
# #     language: str = "python",
# # ) -> List[StructureChunk]:
# #     """
# #     Structure-aware chunking for Python code using the built-in ast module.

# #     Extracts top-level:
# #     - functions
# #     - async functions
# #     - classes

# #     Falls back to no chunks if parsing fails; caller can decide fallback behavior.
# #     """
# #     if not code.strip():
# #         return []

# #     if language.lower() != "python":
# #         return []

# #     try:
# #         tree = ast.parse(code)
# #     except SyntaxError:
# #         return []

# #     lines = code.splitlines()
# #     chunks: List[StructureChunk] = []
# #     chunk_num = 0

# #     for node in tree.body:
# #         if isinstance(node, ast.FunctionDef):
# #             start_line = node.lineno
# #             end_line = _get_end_lineno(node)
# #             text = _safe_get_source_segment(lines, start_line, end_line)
# #             chunks.append(
# #                 StructureChunk(
# #                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
# #                     repo=repo,
# #                     path=path,
# #                     language=language,
# #                     chunk_type="function",
# #                     symbol_name=node.name,
# #                     start_line=start_line,
# #                     end_line=end_line,
# #                     text=text,
# #                 )
# #             )
# #             chunk_num += 1

# #         elif isinstance(node, ast.AsyncFunctionDef):
# #             start_line = node.lineno
# #             end_line = _get_end_lineno(node)
# #             text = _safe_get_source_segment(lines, start_line, end_line)
# #             chunks.append(
# #                 StructureChunk(
# #                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
# #                     repo=repo,
# #                     path=path,
# #                     language=language,
# #                     chunk_type="async_function",
# #                     symbol_name=node.name,
# #                     start_line=start_line,
# #                     end_line=end_line,
# #                     text=text,
# #                 )
# #             )
# #             chunk_num += 1

# #         elif isinstance(node, ast.ClassDef):
# #             start_line = node.lineno
# #             end_line = _get_end_lineno(node)
# #             text = _safe_get_source_segment(lines, start_line, end_line)
# #             chunks.append(
# #                 StructureChunk(
# #                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
# #                     repo=repo,
# #                     path=path,
# #                     language=language,
# #                     chunk_type="class",
# #                     symbol_name=node.name,
# #                     start_line=start_line,
# #                     end_line=end_line,
# #                     text=text,
# #                 )
# #             )
# #             chunk_num += 1

# #     return chunks

# def chunk_code_structure_aware(
#     code: str,
#     repo: str,
#     path: str,
#     language: str = "python",
# ) -> List[StructureChunk]:
#     """
#     Structure-aware chunking for Python code using the built-in ast module.

#     Extracts:
#     - module_header chunk (everything before first top-level def/class)
#     - top-level functions
#     - async functions
#     - classes
#     """
#     if not code.strip():
#         return []

#     if language.lower() != "python":
#         return []

#     try:
#         tree = ast.parse(code)
#     except SyntaxError:
#         return []

#     lines = code.splitlines()
#     chunks: List[StructureChunk] = []
#     chunk_num = 0

#     top_level_nodes = []
#     for node in tree.body:
#         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
#             top_level_nodes.append(node)

#     # 1) module header chunk: everything before the first top-level function/class
#     if top_level_nodes:
#         first_line = min(node.lineno for node in top_level_nodes)
#         if first_line > 1:
#             header_text = _safe_get_source_segment(lines, 1, first_line - 1)
#             if header_text.strip():
#                 chunks.append(
#                     StructureChunk(
#                         chunk_id=f"{repo}:{path}:struct{chunk_num}",
#                         repo=repo,
#                         path=path,
#                         language=language,
#                         chunk_type="module_header",
#                         symbol_name="__module_header__",
#                         start_line=1,
#                         end_line=first_line - 1,
#                         text=header_text,
#                     )
#                 )
#                 chunk_num += 1
#     else:
#         # whole file fallback as one structure chunk if no top-level defs/classes
#         whole_text = "\n".join(lines).rstrip()
#         if whole_text.strip():
#             chunks.append(
#                 StructureChunk(
#                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
#                     repo=repo,
#                     path=path,
#                     language=language,
#                     chunk_type="module_file",
#                     symbol_name="__module_file__",
#                     start_line=1,
#                     end_line=len(lines),
#                     text=whole_text,
#                 )
#             )
#             return chunks

#     # 2) top-level defs/classes
#     for node in tree.body:
#         if isinstance(node, ast.FunctionDef):
#             start_line = node.lineno
#             end_line = _get_end_lineno(node)
#             text = _safe_get_source_segment(lines, start_line, end_line)
#             chunks.append(
#                 StructureChunk(
#                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
#                     repo=repo,
#                     path=path,
#                     language=language,
#                     chunk_type="function",
#                     symbol_name=node.name,
#                     start_line=start_line,
#                     end_line=end_line,
#                     text=text,
#                 )
#             )
#             chunk_num += 1

#         elif isinstance(node, ast.AsyncFunctionDef):
#             start_line = node.lineno
#             end_line = _get_end_lineno(node)
#             text = _safe_get_source_segment(lines, start_line, end_line)
#             chunks.append(
#                 StructureChunk(
#                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
#                     repo=repo,
#                     path=path,
#                     language=language,
#                     chunk_type="async_function",
#                     symbol_name=node.name,
#                     start_line=start_line,
#                     end_line=end_line,
#                     text=text,
#                 )
#             )
#             chunk_num += 1

#         elif isinstance(node, ast.ClassDef):
#             start_line = node.lineno
#             end_line = _get_end_lineno(node)
#             text = _safe_get_source_segment(lines, start_line, end_line)
#             chunks.append(
#                 StructureChunk(
#                     chunk_id=f"{repo}:{path}:struct{chunk_num}",
#                     repo=repo,
#                     path=path,
#                     language=language,
#                     chunk_type="class",
#                     symbol_name=node.name,
#                     start_line=start_line,
#                     end_line=end_line,
#                     text=text,
#                 )
#             )
#             chunk_num += 1

#     return chunks


# def chunk_record_structure_aware(
#     record: Dict[str, Any],
#     code_field: str = "code",
#     repo_field: str = "repo",
#     path_field: str = "path",
#     language_field: str = "language",
#     fallback_to_fixed: bool = True,
#     fixed_chunk_size_tokens: int = 300,
#     fixed_chunk_overlap_tokens: int = 50,
# ) -> List[Dict[str, Any]]:
#     """
#     Chunk one dataset record with structure-aware chunking.
#     Falls back to fixed-size chunking if parsing fails or no structure chunks are found.
#     """
#     code = record.get(code_field, "")
#     repo = record.get(repo_field, "unknown_repo")
#     path = record.get(path_field, "unknown_path")
#     language = record.get(language_field, "python")

#     chunks = chunk_code_structure_aware(
#         code=code,
#         repo=repo,
#         path=path,
#         language=language,
#     )

#     if chunks:
#         return [c.to_dict() for c in chunks]

#     if fallback_to_fixed:
#         fixed_chunks = chunk_code_fixed(
#             code=code,
#             repo=repo,
#             path=path,
#             language=language,
#             chunk_size_tokens=fixed_chunk_size_tokens,
#             chunk_overlap_tokens=fixed_chunk_overlap_tokens,
#         )
#         out = []
#         for c in fixed_chunks:
#             d = c.to_dict()
#             d["chunk_type"] = "fixed_fallback"
#             d["symbol_name"] = ""
#             out.append(d)
#         return out

#     return []


# if __name__ == "__main__":
#     sample_code = """
# import os

# def add(a, b):
#     return a + b

# async def fetch_data():
#     return 42

# class MathOps:
#     def mul(self, a, b):
#         return a * b
# """.strip()

#     record = {
#         "repo": "demo_repo",
#         "path": "math_utils.py",
#         "language": "python",
#         "code": sample_code,
#     }

#     chunks = chunk_record_structure_aware(record)

#     for c in chunks:
#         print("=" * 80)
#         print(f"chunk_id: {c['chunk_id']}")
#         print(f"type: {c['chunk_type']}")
#         print(f"symbol: {c['symbol_name']}")
#         print(f"lines: {c['start_line']}-{c['end_line']}")
#         print(c["text"])


from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import ast

from src.chunking.fixed_chunker import chunk_code_fixed
import textwrap

@dataclass
class StructureChunk:
    chunk_id: str
    repo: str
    path: str
    language: str
    chunk_type: str
    symbol_name: str
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_get_source_segment(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx]).rstrip()


def _get_end_lineno(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return getattr(node, "lineno", 1)
    return end_lineno


def _approx_token_count(text: str) -> int:
    return len(text.split()) if text else 0


def _split_structure_segment(
    text: str,
    repo: str,
    path: str,
    language: str,
    chunk_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    chunk_id_prefix: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> List[StructureChunk]:
    """
    Keep structure-aware boundaries, but split oversized segments into
    smaller fixed-size overlapping subchunks.
    """
    approx_tokens = _approx_token_count(text)

    if approx_tokens <= max_chunk_tokens:
        return [
            StructureChunk(
                chunk_id=chunk_id_prefix,
                repo=repo,
                path=path,
                language=language,
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                start_line=start_line,
                end_line=end_line,
                text=text,
            )
        ]

    subchunks = chunk_code_fixed(
        code=text,
        repo=repo,
        path=path,
        language=language,
        chunk_size_tokens=max_chunk_tokens,
        chunk_overlap_tokens=overlap_tokens,
    )

    out: List[StructureChunk] = []
    for i, sub in enumerate(subchunks):
        adjusted_start = start_line + sub.start_line - 1
        adjusted_end = start_line + sub.end_line - 1
        out.append(
            StructureChunk(
                chunk_id=f"{chunk_id_prefix}_sub{i}",
                repo=repo,
                path=path,
                language=language,
                chunk_type=f"{chunk_type}_subchunk",
                symbol_name=symbol_name,
                start_line=adjusted_start,
                end_line=adjusted_end,
                text=sub.text,
            )
        )

    return out


def chunk_code_structure_aware(
    code: str,
    repo: str,
    path: str,
    language: str = "python",
    max_chunk_tokens: int = 200,
    overlap_tokens: int = 40,
) -> List[StructureChunk]:
    """
    Structure-aware chunking for Python code using the built-in ast module.

    Extracts:
    - module_header chunk
    - top-level functions
    - async functions
    - classes

    Oversized structure chunks are split into smaller subchunks.
    """
    if not code.strip():
        return []

    if language.lower() != "python":
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines()
    chunks: List[StructureChunk] = []
    chunk_num = 0

    top_level_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_nodes.append(node)

    # 1) module header chunk
    if top_level_nodes:
        first_line = min(node.lineno for node in top_level_nodes)
        if first_line > 1:
            header_text = _safe_get_source_segment(lines, 1, first_line - 1)
            if header_text.strip():
                new_chunks = _split_structure_segment(
                    text=header_text,
                    repo=repo,
                    path=path,
                    language=language,
                    chunk_type="module_header",
                    symbol_name="__module_header__",
                    start_line=1,
                    end_line=first_line - 1,
                    chunk_id_prefix=f"{repo}:{path}:struct{chunk_num}",
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
                chunks.extend(new_chunks)
                chunk_num += 1
    else:
        whole_text = "\n".join(lines).rstrip()
        if whole_text.strip():
            new_chunks = _split_structure_segment(
                text=whole_text,
                repo=repo,
                path=path,
                language=language,
                chunk_type="module_file",
                symbol_name="__module_file__",
                start_line=1,
                end_line=len(lines),
                chunk_id_prefix=f"{repo}:{path}:struct{chunk_num}",
                max_chunk_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
            chunks.extend(new_chunks)
            return chunks

    # 2) defs/classes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            chunk_type = "function"
            symbol_name = node.name
        elif isinstance(node, ast.AsyncFunctionDef):
            chunk_type = "async_function"
            symbol_name = node.name
        elif isinstance(node, ast.ClassDef):
            chunk_type = "class"
            symbol_name = node.name
        else:
            continue

        start_line = node.lineno
        end_line = _get_end_lineno(node)
        text = _safe_get_source_segment(lines, start_line, end_line)

        new_chunks = _split_structure_segment(
            text=text,
            repo=repo,
            path=path,
            language=language,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            chunk_id_prefix=f"{repo}:{path}:struct{chunk_num}",
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(new_chunks)
        chunk_num += 1

    return chunks


def chunk_record_structure_aware(
    record: Dict[str, Any],
    code_field: str = "code",
    repo_field: str = "repo",
    path_field: str = "path",
    language_field: str = "language",
    fallback_to_fixed: bool = True,
    fixed_chunk_size_tokens: int = 300,
    fixed_chunk_overlap_tokens: int = 50,
    max_chunk_tokens: int = 200,
    overlap_tokens: int = 40,
) -> List[Dict[str, Any]]:
    code = record.get(code_field, "")
    repo = record.get(repo_field, "unknown_repo")
    path = record.get(path_field, "unknown_path")
    language = record.get(language_field, "python")

    chunks = chunk_code_structure_aware(
        code=code,
        repo=repo,
        path=path,
        language=language,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    if chunks:
        return [c.to_dict() for c in chunks]

    if fallback_to_fixed:
        fixed_chunks = chunk_code_fixed(
            code=code,
            repo=repo,
            path=path,
            language=language,
            chunk_size_tokens=fixed_chunk_size_tokens,
            chunk_overlap_tokens=fixed_chunk_overlap_tokens,
        )
        out = []
        for c in fixed_chunks:
            d = c.to_dict()
            d["chunk_type"] = "fixed_fallback"
            d["symbol_name"] = ""
            out.append(d)
        return out

    return []


if __name__ == "__main__":
    sample_code = textwrap.dedent("""
    import os

    def add(a, b):
        return a + b

    async def fetch_data():
        return 42

    class MathOps:
        def mul(self, a, b):
            return a * b

    def very_long_function():
        x = 1
        y = 2
        z = 3
        a = 4
        b = 5
        c = 6
        d = 7
        e = 8
        f = 9
        g = 10
        h = 11
        i = 12
        return x + y + z + a + b + c + d + e + f + g + h + i
    """).strip()

    record = {
        "repo": "demo_repo",
        "path": "math_utils.py",
        "language": "python",
        "code": sample_code,
    }

    chunks = chunk_record_structure_aware(
        record,
        max_chunk_tokens=30,
        overlap_tokens=10,
    )

    for c in chunks:
        print("=" * 80)
        print(f"chunk_id: {c['chunk_id']}")
        print(f"type: {c['chunk_type']}")
        print(f"symbol: {c['symbol_name']}")
        print(f"lines: {c['start_line']}-{c['end_line']}")
        print(c["text"])