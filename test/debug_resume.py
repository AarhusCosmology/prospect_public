import os
import shutil
from prospect.run import run
test_input = "test/debug_resume/resume_test"
output_base = "output/debug_resume"

output_idx = 0
test_output = f"{output_base}_{output_idx}"

while True:
    if os.path.isdir(test_output):
        output_idx += 1
    else:
        break
    test_output = f"{output_base}_{output_idx}"

shutil.copytree(test_input, test_output)
run(test_output)
