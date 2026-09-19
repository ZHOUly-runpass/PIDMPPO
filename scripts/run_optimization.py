"""One bounded, auditable optimization round. Never dispatches the 65-job matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml

from pidmppo.evaluation.optimization import confirmation_gate, select_candidate, training_health
from pidmppo.evaluation.protocol import load_scenarios
from pidmppo.utils.manifest import source_fingerprint


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class BudgetStopped(RuntimeError):
    pass


class Round:
    def __init__(self, spec, output: Path, max_hours=None, max_jobs=None):
        self.spec, self.output = spec, output
        self.started = time.time()
        self.deadline = self.started + min(spec["max_hours"], max_hours or spec["max_hours"]) * 3600
        self.max_jobs = min(spec["max_train_jobs"], max_jobs if max_jobs is not None else spec["max_train_jobs"])
        self.train_jobs = 0
        self.stop = False
        self.state = {"version": spec["version"], "status": "running", "started_unix": self.started,
                      "deadline_unix": self.deadline, "pid": os.getpid(), "max_train_jobs": self.max_jobs,
                      "jobs": [], "screening": {}, "confirmation": {}, "legacy": {}, "formal_training_dispatched": False}

    def save(self):
        self.state["updated_unix"] = time.time()
        self.state["elapsed_seconds"] = time.time() - self.started
        self.state["train_jobs_dispatched"] = self.train_jobs
        atomic_json(self.output / "status.json", self.state)
        lines = ["# Correctness-v2 optimization status", "", f"Status: {self.state['status']}; backend: grid (not PyBullet reproduction).", "",
                 "All old runs are retained. Screening uses only training-distribution validation.", "",
                 "| Candidate | Success | Collision | Successful path efficiency |", "|---|---:|---:|---:|"]
        for key, result in self.state["screening"].items():
            lines.append(f"| {key} | {result['success_rate']:.1%} | {result['collision_rate']:.1%} | {result['successful_path_efficiency']:.3f} |")
        lines += ["", "## Legacy checkpoint re-evaluation (new fixed scenes)", "", "| Seed | Success | Collision |", "|---|---:|---:|"]
        for key, result in self.state["legacy"].items():
            lines.append(f"| {key} | {result['success_rate']:.1%} | {result['collision_rate']:.1%} |")
        if "gate" in self.state:
            lines.extend(["", "## Gate", "", "```json", json.dumps(self.state["gate"], ensure_ascii=False, indent=2), "```"])
        lines.extend(["", "Pending cells are not results. No final-test scores are used for candidate selection.",
                      "Training-health reports and representative outcome traces are saved next to each job."])
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_command(self, command, name, kind):
        if self.stop or time.time() >= self.deadline - self.spec["dispatch_reserve_seconds"]:
            raise BudgetStopped("Deadline reserve reached; no new tasks dispatched")
        if kind == "train":
            if self.train_jobs >= self.max_jobs:
                raise BudgetStopped("Explicit training-job limit reached")
            self.train_jobs += 1
            command += ["--deadline-unix", str(self.deadline - 60)]
        record = {"name": name, "kind": kind, "command": command, "status": "running", "started_unix": time.time()}
        self.state["jobs"].append(record)
        self.save()
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        print(f"[{kind}] {name}: {shlex.join(command)}", flush=True)
        try:
            with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                sent_stop = False
                while child.poll() is None:
                    if (self.stop or time.time() >= self.deadline - 60) and not sent_stop:
                        child.terminate()
                        sent_stop = True
                        record["stop_requested_unix"] = time.time()
                        self.save()
                    # A train child checkpoints at a rollout/minibatch boundary.
                    # Do not kill it merely because this checkpoint takes time.
                    time.sleep(1)
                record["returncode"] = child.returncode
                if sent_stop:
                    record["status"] = "interrupted"
                    raise BudgetStopped("Budget/signal stop requested; current task not treated as completed")
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
            record["status"] = "completed"
        except Exception:
            if record["status"] == "running":
                record["status"] = "failed"
            raise
        finally:
            record["ended_unix"] = time.time()
            self.save()

    def evaluate(self, checkpoint, name, seed, suite, variant="pidmppo"):
        output = self.output / "evaluation" / name
        command = [sys.executable, "scripts/evaluate.py", str(checkpoint), "--variant", variant,
                   "--scenarios", str(Path(self.spec["assets"]) / f"{suite}.json"), "--training-seed", str(seed),
                   "--trace-outcomes", "--output", str(output)]
        self.run_command(command, f"eval_{name}", "evaluation")
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))

    def train_and_validate(self, candidate, seed, steps, phase, variant="pidmppo"):
        name = f"{phase}_{candidate}_{variant}_{seed}"
        output = self.output / "training" / name
        command = [sys.executable, "scripts/train.py", "--config", self.spec["base_config"], "--seed", str(seed),
                   "--steps", str(steps), "--variant", variant, "--output", str(output)]
        for value in self.spec["candidates"][candidate]:
            command.extend(("--set", value))
        self.run_command(command, name, "train")
        status = json.loads((output / "train_status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            raise BudgetStopped(f"Train job {name} stopped before completion")
        health = training_health(output / "training_metrics.csv")
        atomic_json(output / "health.json", health)
        return self.evaluate(output / "checkpoints/final.pt", name, seed, "validation", variant)

    def execute(self):
        for seed in self.spec["legacy_seeds"]:
            checkpoint = Path(self.spec["legacy_root"]) / f"seed_{seed}" / "checkpoints/final.pt"
            self.state["legacy"][str(seed)] = self.evaluate(checkpoint, f"legacy_{seed}", seed, "fixed")
            self.save()
        for candidate in ("A", "B", "C", "D"):
            self.state["screening"][candidate] = self.train_and_validate(candidate, self.spec["screen_seed"], self.spec["screen_steps"], "screen")
            self.save()
        selected = select_candidate(self.state["screening"])
        self.state["selected"] = selected
        if selected is None:
            self.state["status"] = "hold_no_screening_gain"
            self.state["gate"] = {"status": "hold", "reason": "No B/C/D exceeds A by at least 5 percentage points", "formal_training_dispatched": False}
            self.save()
            return
        for candidate in ("A", selected):
            results = []
            self.state["confirmation"][candidate] = results
            for seed in self.spec["confirmation_seeds"]:
                results.append(self.train_and_validate(candidate, seed, self.spec["confirmation_steps"], "confirm"))
                self.save()
        self.state["gru_diagnostic"] = self.train_and_validate(selected, 11, self.spec["confirmation_steps"], "diagnostic", "gru_ppo")
        self.state["gate"] = confirmation_gate(self.state["confirmation"]["A"], self.state["confirmation"][selected])
        self.state["status"] = "completed_eligible_for_review" if self.state["gate"]["status"] == "eligible_for_review" else "completed_hold"
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/optimization_v2.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/optimization_v2"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-train-jobs", type=int)
    parser.add_argument("--preflight", type=Path, default=Path("artifacts/correctness_v2_preflight.json"))
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_hours is not None and args.max_hours <= 0 or args.max_train_jobs is not None and args.max_train_jobs < 0:
        raise ValueError("Budget must be positive and job limit nonnegative")
    print(json.dumps({"screen": list(spec["candidates"]), "nominal_max_training_steps": 4 * spec["screen_steps"] + 7 * spec["confirmation_steps"],
                      "max_train_jobs": spec["max_train_jobs"], "max_hours": spec["max_hours"], "execute": args.execute}, indent=2))
    if not args.execute:
        return
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("version") != spec["version"]:
        raise ValueError("Correctness-v2 preflight has not passed")
    if preflight.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Implementation/protocol changed after preflight; run a fresh preflight")
    for name, expected in (("fixed", 400), ("validation", 100), ("random", 60)):
        path = Path(spec["assets"]) / f"{name}.json"
        scenes = load_scenarios(path)
        if len(scenes) != expected or any(s.version != 2 for s in scenes):
            raise ValueError(f"Incorrect scene count/version: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != preflight["manifest_hashes"][name]:
            raise ValueError("Preflight asset hashes do not match current scenes")
    for seed in spec["legacy_seeds"]:
        if not (Path(spec["legacy_root"]) / f"seed_{seed}" / "checkpoints/final.pt").is_file():
            raise FileNotFoundError(f"Missing old checkpoint seed {seed}")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output; a stopped round cannot silently reset its budget: {args.output}")
    args.output.mkdir(parents=True)
    runner = Round(spec, args.output, args.max_hours, args.max_train_jobs)
    signal.signal(signal.SIGTERM, lambda *_: setattr(runner, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(runner, "stop", True))
    runner.save()
    try:
        runner.execute()
    except BudgetStopped as error:
        runner.state.update(status="stopped_budget_or_signal", reason=str(error))
    except Exception as error:
        runner.state.update(status="failed", reason=repr(error))
        raise
    finally:
        runner.save()


if __name__ == "__main__":
    main()
