from unittest import TestCase
import tempfile
from pathlib import Path
import shutil
from argparse import Namespace
import filecmp
import os.path

from pythontemplatepackage.rename import main


class TestRename(TestCase):

    # from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
    def assertDirsEqual(self, dir1, dir2):
        dirs_cmp = filecmp.dircmp(dir1, dir2)
        if len(dirs_cmp.left_only) > 0:
            raise AssertionError(f"Some files/directories are only in {dir1}: {dirs_cmp.left_only}")
        if len(dirs_cmp.right_only) > 0:
            raise AssertionError(f"Some files/directories are only in {dir2}: {dirs_cmp.right_only}")
        if len(dirs_cmp.funny_files) > 0:
            raise AssertionError(f"Some files/directories are 'funny': {dirs_cmp.funny_files}")
        (_, mismatch, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
        if len(mismatch) > 0:
            raise AssertionError(f"Some files in {dir1} and {dir2} are not identical: {mismatch}")
        if len(errors) > 0:
            raise AssertionError(f"There were errors comparing some files in {dir1} and {dir2}: {errors}")
        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            self.assertDirsEqual(new_dir1, new_dir2)

    def test_rename(self):

        class Ignore:
            def __init__(self, base):
                self.base = base

            def __call__(self, dir, sub):
                if Path(dir) == Path(self.base):
                    return [s for s in sub if s not in ['pythontemplatepackage',
                                                        'doc',
                                                        '.github',
                                                        '.gitignore',
                                                        'README.md',
                                                        'setup.py',
                                                        'LICENSE',
                                                        'tests',
                                                        'requirements.txt',
                                                        'examples']]
                if Path(dir) == Path(self.base) / 'doc':
                    return [s for s in sub if s not in ['api_summary.rst',
                                                        'conf.py',
                                                        'index.rst',
                                                        'requirements.txt',
                                                        '_static',
                                                        '_templates']]
                return []

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = Path(__file__).parent.resolve() / '..'  # source directory
            cmp_dir = Path(__file__).parent.resolve() / '..' / '.rename_test_dir'  # directory to compare against
            # tmp_dir = cmp_dir  # uncomment to generate test files
            shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=True, ignore=Ignore(src_dir))
            main(Namespace(dir=tmp_dir,
                           apply=True,
                           # apply=False,
                           package_name='NewPackage',
                           python_name='newpackage',
                           author='<New Author>',
                           author_email='<new.author.email>',
                           copyright='<newcopyright>',
                           github='<newgithuburl>',
                           github_doc='<newgithubdocurl>',
                           codecov='<newcodecovurl>'))
            self.assertDirsEqual(tmp_dir, cmp_dir)
