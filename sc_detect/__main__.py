#!/usr/bin/env python
#
#         Pysc_detect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/pysc_detect/   ]
#     [  Github: https://github.com/Breakthrough/Pysc_detect/  ]
#     [  Documentation: http://pysc_detect.readthedocs.org/    ]
#
# Provides functionality to run Pysc_detect directly as a Python module (in
# addition to using in other scripts via `import sc_detect`) by running:
#
#   > python -m sc_detect
#
# Installing Pysc_detect (using `python setup.py install` in the parent
# directory) will also add the `sc_detect` command to be used from anywhere,
# e.g. `sc_detect -i myfile.mp4`.
#
#
# Copyright (C) 2012-2017 Brandon Castellano <http://www.bcastell.com>.
#
# Pysc_detect is licensed under the BSD 2-Clause License; see the
# included LICENSE file or visit one of the following pages for details:
#  - http://www.bcastell.com/projects/pysc_detect/
#  - https://github.com/Breakthrough/Pysc_detect/
#
# This software uses Numpy and OpenCV; see the LICENSE-NUMPY and
# LICENSE-OPENCV files or visit one of above URLs for details.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

if __name__ == '__main__':
    import sc_detect
    sc_detect.main()

