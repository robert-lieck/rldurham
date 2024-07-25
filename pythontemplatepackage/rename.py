#  Copyright (c) 2024 Robert Lieck.
import argparse
import os
from pathlib import Path

from colorama import Fore, Back, Style, init as colorama_init
colorama_init()


def replace_in_file(file_path, find_str, replace_str, dry_run=True, print_content=True):
    # print info
    print(f"{Fore.GREEN}In '{file_path}', replacing '{find_str}' with '{replace_str}'{Style.RESET_ALL}")
    # get full path
    file_path = Path(os.getcwd()) / Path(file_path)
    # read file content / catch file not found errors
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
    except FileNotFoundError:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR: The file '{file_path}' was not found, please double-check your file structure.{Style.RESET_ALL}")
        exit(1)
    # highlight replacement for print-only runs
    if dry_run and print_content:
        replace_str = Style.BRIGHT + Fore.YELLOW + replace_str + Style.RESET_ALL
    # replace
    file_contents = file_contents.replace(find_str, replace_str)
    # print content
    if print_content:
        print(file_contents)
    # overwrite file
    if not dry_run:
        with open(file_path, 'w') as file:
            file.write(file_contents)


def main(args):
    print(args)
    # change working directory and print message
    cwd = os.getcwd()
    os.chdir(args.dir)
    print(f"{Fore.GREEN}Working directory: {os.getcwd()}{Style.RESET_ALL}")

    # dry run message
    if not args.apply:
        print(f"{Style.BRIGHT}{Fore.YELLOW}Performing dry run...{Style.RESET_ALL}")

    # replace strings in files
    for old, new, prompt, file_list in [
        ('PythonTemplatePackage', args.package_name, 'Package name in text', ['./doc/index.rst',
                                                                              './README.md',
                                                                              './examples/README.rst']),
        ('pythontemplatepackage', args.python_name, 'Python package name', ['./doc/conf.py',
                                                                            './doc/index.rst',
                                                                            './doc/api_summary.rst',
                                                                            './.github/workflows/tests.yml',
                                                                            './.github/workflows/test_dev.yml',
                                                                            './README.md',
                                                                            './setup.py',
                                                                            './tests/test_template.py']),
        ('Robert Lieck', args.author, 'Author', ['./doc/conf.py',
                                                 './setup.py']),
        ('robert.lieck@durham.ac.uk', args.author_email, 'Author email', ['./setup.py']),
        ('2022 Robert Lieck', args.copyright, 'Copyright', ['./doc/conf.py',
                                                            './tests/test_examples.py']),
        ('https://github.com/robert-lieck', args.github, 'GitHub base URL', ['./README.md',
                                                                             './setup.py']),
        ('https://robert-lieck.github.io', args.github_doc, 'GitHub doc base URL', ['./README.md']),
        ('https://codecov.io/gh/robert-lieck', args.codecov, 'Codecov base URL', ['./README.md']),
    ]:
        # prompt user for input if not provided as command line argument
        if new is None:
            new = input(f"{Style.BRIGHT}{Fore.GREEN}{prompt}\n"
                        f"{Style.NORMAL}replaces '{old}' in {file_list}\n"
                        f"{Style.BRIGHT}[ENTER to skip]: {Style.RESET_ALL}")
        # skip if empty
        if new:
            # replace in files
            for f in file_list:
                replace_in_file(f, old, new, dry_run=not args.apply, print_content=not args.apply)
            # rename package folder
            if old == 'pythontemplatepackage':
                print(f"{Fore.GREEN}Rename directory '{old}' to '{new}'{Style.RESET_ALL}")
                if not os.path.isdir(old):
                    print(f"{Fore.RED}{Style.BRIGHT}ERROR: '{old}' is not a directory and cannot be renamed, please double-check your file structure.{Style.RESET_ALL}")
                    exit(1)
                if args.apply:
                    os.rename(old, new)
        else:
            print(f"{Style.BRIGHT}{Fore.YELLOW}Skipping!{Style.RESET_ALL}")
    # change back to original working directory
    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to adapt the template for a specific Python package")
    parser.add_argument('-a', '--apply', action='store_true', help='apply changes instead of just doing a dry run')
    parser.add_argument('-d', '--dir', type=str, default=".", help='root directory (parent directory of the new package); defaults to current working directory')

    parser.add_argument('--package_name', type=str, default=None, help="Package name in text (replaces: 'PythonTemplatePackage')")
    parser.add_argument('--python_name', type=str, default=None, help="Python package name (replaces: 'pythontemplatepackage')")
    parser.add_argument('--author', type=str, default=None, help="Author (replaces: 'Robert Lieck')")
    parser.add_argument('--author_email', type=str, default=None, help="Author email (replaces: 'robert.lieck@durham.ac.uk')")
    parser.add_argument('--copyright', type=str, default=None, help="Copyright (replaces: '2022 Robert Lieck')")
    parser.add_argument('--github', type=str, default=None, help="GitHub base URL (replaces: 'https://github.com/robert-lieck')")
    parser.add_argument('--github_doc', type=str, default=None, help="GitHub doc base URL (replaces: 'https://robert-lieck.github.io')")
    parser.add_argument('--codecov', type=str, default=None, help="Codecov base URL (replaces: 'https://codecov.io/gh/robert-lieck')")

    args = parser.parse_args()

    main(args)
