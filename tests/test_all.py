# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run all tests

import os
import sys
import argparse
import importlib
import time
import traceback
from pomdp_py.utils import typ

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Running tests.")
    args = parser.parse_args()

    # load the test modules
    tests = []
    for fname in sorted(os.listdir(ABS_DIR)):
        if fname != "test_all.py" and fname.startswith("test") and fname.endswith(".py"):
            test_module = importlib.import_module(fname.split(".py")[0])
            tests.append(test_module)

    for i, test_module in enumerate(tests):
        print(typ.bold("[{}/{}] {}".format(i+1, len(tests), test_module.description)))

        old_stdout = sys.stdout
        try:
            test_module.run()
        except Exception as ex:
            sys.stdout = old_stdout
            print(typ.red(traceback.format_exc()))

if __name__ == "__main__":
    main()
