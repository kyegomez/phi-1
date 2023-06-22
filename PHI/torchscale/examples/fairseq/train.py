# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# flake8: noqa
import models
import tasks
import criterions
from fairseq_cli.train import cli_main

if __name__ == "__main__":
    cli_main()
