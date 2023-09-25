import re
import torch
import numpy
import random
import logging

from typing import *


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def is_code_generation_finished(
    code: str,
    dataset_type: str = None,
    language_type: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if dataset_type == "mbpp":
        end_words = ["\ndef", "\nassert"]
        for w in end_words:
            if w == "\ndef":
                if code.count(w) > 1:
                    return True
            else:
                if w in code:
                    return True
    else:
        if language_type.lower() == "python":
            code_splits = code.split("\n")
            for i, line in enumerate(code_splits):
                if (
                    len(line.strip()) > 0
                    and line[0] != " "
                    and line[0] != "\t"
                    and i > 0
                ):
                    return True
            end_words = [
                "\ndef",
                "\nclass",
                "\nif",
                "\n#",
                "\nprint",
                "\nassert",
                '\n"""',
                "\n\n\n",
            ]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if "\nfunc main(" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if "\nint main()" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "rust":
            if "\nfn main()" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True

    return False


# Modified from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/lm_eval/tasks/mbpp.py
stop_words = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]


def first_block(string, stop_words):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(stop_words), string)[0].rstrip()


def cleanup_code(
    code: str,
    dataset_type: str = None,
    language_type: str = None,
):
    """
    Cleans up the generated code.
    """
    if dataset_type == "mbpp":
        end_words = ["\nassert", "\ndef"]
        for w in end_words:
            if w == "\ndef":
                if code.count(w) > 1:
                    code = code[: code.rfind(w)]
            else:
                code = code[: code.rfind(w)]
        code = first_block(code, stop_words)
    elif dataset_type == "humanevalx":
        if language_type.lower() == "python":
            code_splits = code.split("\n")
            is_empty_line = False
            ind_empty_line = None
            for i, line in enumerate(code_splits):
                if (
                    len(line.strip()) > 0
                    and line[0] != " "
                    and line[0] != "\t"
                    and i > 0
                ):
                    is_empty_line = True
                    ind_empty_line = i
                    break
            if is_empty_line:
                code = "\n".join(code_splits[:ind_empty_line])
            else:
                end_words = [
                    "\ndef",
                    "\nclass",
                    "\n#",
                    "\nassert",
                    '\n"""',
                    "\nprint",
                    "\nif",
                    "\n\n\n",
                ]
                for w in end_words:
                    if w in code:
                        code = code[: code.rfind(w)]
        elif language_type.lower() == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + "}"
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if code.count("{") + 1 == code.count("}"):
                code += "\n}"
        elif language_type.lower() == "go":
            if "\nfunc main(" in code:
                code = code[: code.rfind("func main(")]
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
        elif language_type.lower() == "cpp":
            if "\nint main()" in code:
                code = code[: code.rfind("int main()")]
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
        elif language_type.lower() == "js":
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
        elif language_type.lower() == "rust":
            if "}" in code:
                code = code[: code.rfind("}")] + "}"

    return code


class Logger:
    def __init__(
        self,
        name,
        log_level=logging.INFO,
        log_file=None,
        log_mode="both",
        disable_formatter=False,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        self.formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Log to console
        if log_mode == "both" or log_mode == "terminal":
            console_handler = logging.StreamHandler()
            if not disable_formatter:
                console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        # Log to file
        if log_file is not None:
            if log_mode == "both" or log_mode == "file":
                file_handler = logging.FileHandler(log_file, mode="w")
                if not disable_formatter:
                    file_handler.setFormatter(self.formatter)
                self.logger.addHandler(file_handler)

    def add_file_handler(self, file_name):
        file_handler = logging.FileHandler(file_name, mode="w")
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
