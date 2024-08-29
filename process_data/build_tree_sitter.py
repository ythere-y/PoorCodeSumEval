from tree_sitter import Language
import os

print(os.getcwd())
Language.build_library(
    # Store the library in the `build` directory
    "process_data/build/my-languages.so",
    # Include one or more languages
    [
        "process_data/vendor/tree-sitter-go",
        "process_data/vendor/tree-sitter-java",
        "process_data/vendor/tree-sitter-python",
    ],
)
