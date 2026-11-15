"""Fetch finished wandb runs for the figures in scripts/."""

from collections import defaultdict

import wandb


def latest_runs(*keys: str) -> dict[tuple[str, str], list]:
    """(experiment, run name) -> the seeds of its latest Slurm job whose summary has all `keys`.

    Run ids are "<exp>-<job>-<run>" (scripts/common.sh). Runs trained before the cosine
    learning-rate decay are tagged "constant-lr" in wandb and skipped.
    """
    api = wandb.Api()
    jobs = defaultdict(lambda: defaultdict(list))
    for run in api.runs(f"{api.default_entity}/bubblelat"):
        if (run.state == "finished" and all(k in run.summary for k in keys)
                and "constant-lr" not in run.tags):
            jobs[run.group, run.name][run.id.split("-")[1]].append(run)
    return {key: by_job[max(by_job, key=int)] for key, by_job in jobs.items()}
