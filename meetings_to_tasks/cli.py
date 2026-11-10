"""m2t: one entry point for every step of a job.

  m2t run <files...> --out results/<job>   transcribe, separate voices, write minutes and tasks
  m2t names <job> <names.json>           apply confirmed speaker names
  m2t tasks <job>                        re-export tasks after editing summary.md
  m2t voices enroll|match <job>          local library of confirmed voices
  m2t verify <job> [recordings]          completeness and timestamp checks

Run "m2t <command> --help" for the options of a command.
"""
import importlib
import sys

# Modules load on demand: only "run" needs torch and the speech models.
COMMANDS = {'run': 'pipeline', 'names': 'names', 'tasks': 'tasks', 'voices': 'voices', 'verify': 'verify'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ('-h', '--help') else 2)
    module = importlib.import_module(f'.{COMMANDS[argv[0]]}', __package__)
    module.main(argv[1:])
