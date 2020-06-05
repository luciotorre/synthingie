from subprocess import run, PIPE

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

git_cmd = "git ls-files --error-unmatch".split(" ")


def is_git_tracked(filename):
    result = run(git_cmd + [filename], stdout=PIPE, stderr=PIPE)

    return result.returncode == 0


def pytest_collect_file(parent, path):
    if path.ext == ".ipynb" and is_git_tracked(path):
        return IPYNBFile.from_parent(parent, fspath=path)


class IPYNBFile(pytest.File):
    def collect(self):
        return [IPYNBItem.from_parent(
            name=str(self.fspath),
            parent=self,
            filename=self.fspath
        )]


class IPYNBItem(pytest.Item):
    def __init__(self, name, parent, filename):
        super().__init__(name, parent)
        self.filename = filename

    def runtest(self):
        with open(self.filename) as f:
            nb = nbformat.read(f, as_version=4)

        run_path = self.filename.dirname
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': run_path}})

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        return str(excinfo._excinfo[1])

    def reportinfo(self):
        return self.filename, 0, self.name
