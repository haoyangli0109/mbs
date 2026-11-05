"""Start the ATOM OpenAI server; same arguments as `python -m atom.entrypoints.openai_server`.

Another session on this machine kills every process matching
"python -m atom.entrypoints" before each of its runs, so our servers must not
carry that string in their command line.
"""

import runpy

runpy.run_module("atom.entrypoints.openai_server", run_name="__main__", alter_sys=True)
