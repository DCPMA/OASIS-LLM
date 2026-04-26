"""CLI entry point."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import RunConfig
from .db import connect
from .enqueue import enqueue_trials, upsert_run
from .runner import run as run_async

app = typer.Typer(add_completion=False, help="OASIS-LLM: rate OASIS images with LLMs.")
dataset_app = typer.Typer(help="Manage curated image datasets.")
experiment_app = typer.Typer(help="Manage multi-config experiments.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")
console = Console()


@app.command()
def run(config: Path = typer.Argument(..., help="Path to run YAML config")):
    """Enqueue (idempotent) and execute a run."""
    cfg = RunConfig.from_yaml(config)
    con = connect()
    run_id, created = upsert_run(con, cfg)
    console.print(f"[bold]{'Created' if created else 'Resuming'}[/] run [cyan]{run_id}[/]")
    inserted = enqueue_trials(con, cfg)
    console.print(f"Enqueued [yellow]{inserted}[/] new trials")
    pending = con.execute(
        "SELECT count(*) FROM trials WHERE run_id=? AND status IN ('pending','failed')",
        [run_id],
    ).fetchone()[0]
    console.print(f"Pending: {pending}")
    if pending == 0:
        console.print("[green]All trials done.[/]")
        return
    asyncio.run(run_async(cfg, con))


@app.command("run-id")
def run_id_cmd(
    run_id: str = typer.Argument(..., help="run_id of a row in the runs table."),
):
    """Execute a run by id, loading its config from the database.

    Used by the scheduler daemon to launch experiment-created runs that
    don't have a YAML file on disk. Resets failed trials → pending so that
    queued reruns retry the failures.
    """
    import json
    con = connect()
    row = con.execute("SELECT config_json FROM runs WHERE run_id=?", [run_id]).fetchone()
    if row is None:
        console.print(f"[red]No run {run_id}[/]")
        raise typer.Exit(1)
    payload = json.loads(row[0])
    payload["name"] = run_id
    cfg = RunConfig(**payload)
    # Reset any failed trials so they get retried.
    con.execute(
        "UPDATE trials SET status='pending', error=NULL "
        "WHERE run_id=? AND status='failed'",
        [run_id],
    )
    pending = con.execute(
        "SELECT count(*) FROM trials WHERE run_id=? AND status IN ('pending','failed')",
        [run_id],
    ).fetchone()[0]
    console.print(f"[bold]Run[/] [cyan]{run_id}[/] — {pending} pending trial(s)")
    if pending == 0:
        console.print("[green]Nothing to do.[/]")
        return
    asyncio.run(run_async(cfg, con))


@app.command()
def scheduler(
    poll_interval: float = typer.Option(5.0, help="Seconds between polls."),
    max_parallel: int | None = typer.Option(
        None, help="Override max_parallel; otherwise read from DB."
    ),
):
    """Run the queue scheduler daemon. Blocks until SIGTERM/SIGINT.

    Spawns queued runs (status='queued') as subprocesses, up to ``max_parallel``
    in flight at once. Configure via Settings page or `--max-parallel`.
    """
    from . import queue as _q
    from . import scheduler as _sch
    con = connect()
    if max_parallel is not None:
        _q.set_max_parallel(con, max_parallel)
    cap = _q.max_parallel(con)
    console.print(
        f"[bold]Scheduler[/] starting · max_parallel={cap} · poll={poll_interval}s"
    )
    con.close()
    _sch.run_daemon(poll_interval_s=poll_interval)



@app.command()
def status(run_id: str | None = typer.Argument(None)):
    """Show run status."""
    con = connect()
    if run_id:
        rows = con.execute(
            "SELECT status, count(*) FROM trials WHERE run_id=? GROUP BY status", [run_id]
        ).fetchall()
        t = Table(title=f"Run {run_id}")
        t.add_column("Status"); t.add_column("Count", justify="right")
        for s, c in rows: t.add_row(s, str(c))
        console.print(t)
    else:
        rows = con.execute(
            """
            SELECT r.run_id, r.status,
                   sum(CASE WHEN t.status='done' THEN 1 ELSE 0 END) AS done,
                   sum(CASE WHEN t.status='pending' THEN 1 ELSE 0 END) AS pending,
                   sum(CASE WHEN t.status='failed' THEN 1 ELSE 0 END) AS failed,
                   round(sum(t.cost_usd), 4) AS cost
            FROM runs r LEFT JOIN trials t USING (run_id)
            GROUP BY r.run_id, r.status
            ORDER BY r.run_id
            """
        ).fetchall()
        t = Table(title="Runs")
        for col in ("run_id", "status", "done", "pending", "failed", "cost_usd"):
            t.add_column(col)
        for r in rows:
            t.add_row(*[str(x) if x is not None else "-" for x in r])
        console.print(t)


@app.command()
def export(run_id: str, out: Path = Path("outputs/llm_trials.csv")):
    """Export a run's completed trials to CSV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.execute(
        f"COPY (SELECT * FROM trials WHERE run_id=? AND status='done') TO '{out}' (HEADER, DELIMITER ',')",
        [run_id],
    )
    console.print(f"[green]Wrote {out}[/]")


@app.command("paper-plots")
def paper_plots(
    run_id: str,
    out_dir: Path = typer.Option(
        Path("outputs/paper_plots"),
        help="Directory where paper-style plots and summaries will be written.",
    ),
):
    """Write paper-style summary plots for a run."""
    from .analysis import export_paper_plots

    target_dir = out_dir / run_id
    summary = export_paper_plots(run_id=run_id, out_dir=target_dir)
    console.print(f"[green]Wrote paper-style plots to {target_dir}[/]")
    console.print(
        "Images={images} | valence samples/image={valence_samples} | "
        "arousal samples/image={arousal_samples} | valence-arousal r={corr}".format(
            images=summary["image_count"],
            valence_samples=summary["valence_samples_per_image"],
            arousal_samples=summary["arousal_samples_per_image"],
            corr=summary["valence_arousal_correlation"],
        )
    )
    if summary["human_valence_correlation"] is not None:
        console.print(
            "Human overlap correlations: valence={valence_corr}, arousal={arousal_corr}".format(
                valence_corr=summary["human_valence_correlation"],
                arousal_corr=summary["human_arousal_correlation"],
            )
        )


@app.command("participant-dataset")
def participant_dataset(
    run_id: str,
    out_dir: Path = typer.Option(
        Path("outputs/participant_dataset"),
        help="Directory where the participant-style dataset and plots will be written.",
    ),
    images_per_participant: int = typer.Option(
        20,
        min=1,
        help="Number of images to include per pseudo-participant row.",
    ),
):
    """Write a wide participant-style CSV and per-picture distribution plots."""
    from .analysis import export_participant_style_dataset

    target_dir = out_dir / run_id
    summary = export_participant_style_dataset(
        run_id=run_id,
        out_dir=target_dir,
        images_per_participant=images_per_participant,
    )
    console.print(f"[green]Wrote attempt-style dataset to {target_dir}[/]")
    console.print(
        "Provider={provider} | model={model_id} | attempts={attempts} | images/attempt={images} | response columns={responses}".format(
            provider=summary["provider"],
            model_id=summary["model_id"],
            attempts=summary["attempt_count"],
            images=summary["images_per_attempt"],
            responses=summary["response_columns"],
        )
    )
    console.print("Rows are reconstructed from sample_idx, not real participant sessions.")


@app.command()
def smoke(
    config: Path = typer.Argument(..., help="Path to run YAML config"),
    n: int = typer.Option(3, help="Number of trials to attempt"),
):
    """Run a smoke test against the given config: 3 images, valence only."""
    cfg = RunConfig.from_yaml(config)
    cfg.image_set = "smoke_3"
    cfg.dimensions = ["valence"]
    cfg.samples_per_image = 1
    cfg.max_concurrency = min(cfg.max_concurrency, 2)
    if not cfg.name.startswith("smoke-"):
        cfg.name = f"smoke-{cfg.name}"
    con = connect()
    upsert_run(con, cfg)
    inserted = enqueue_trials(con, cfg)
    console.print(f"Smoke run [cyan]{cfg.name}[/] enqueued {inserted} trials")
    asyncio.run(run_async(cfg, con))
    rows = con.execute(
        "SELECT image_id, rating, error, latency_ms FROM trials WHERE run_id=?",
        [cfg.name],
    ).fetchall()
    t = Table(title="Smoke results")
    for col in ("image_id", "rating", "error", "latency_ms"):
        t.add_column(col)
    for r in rows:
        t.add_row(*[str(x) if x is not None else "-" for x in r])
    console.print(t)


@app.command()
def dashboard(port: int = 8501):
    """Launch the Streamlit dashboard."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(Path(__file__).parent / "dashboard.py"),
                    "--server.port", str(port)])


# ---------------------------------------------------------------------------
# Dataset subcommands
# ---------------------------------------------------------------------------

@dataset_app.command("list")
def dataset_list():
    """List all datasets."""
    from . import datasets as ds
    con = connect()
    rows = ds.list_all(con)
    t = Table(title="Datasets")
    for col in ("id", "name", "status", "source", "active/total", "created"):
        t.add_column(col)
    for d in rows:
        t.add_row(
            d.dataset_id, d.name, d.status, d.source,
            f"{d.active_count}/{d.image_count}",
            str(d.created_at) if d.created_at else "-",
        )
    console.print(t)


@dataset_app.command("generate")
def dataset_generate(
    name: str = typer.Argument(..., help="Human-readable name (slugified for the id)."),
    n: int = typer.Option(30, help="Target image count."),
    strategy: str = typer.Option("stratified", help="stratified|uniform|all"),
    seed: int = typer.Option(42, help="Sampling seed."),
    description: str | None = typer.Option(None, help="Optional description."),
):
    """Generate a draft dataset by sampling from the full pool."""
    from . import datasets as ds
    con = connect()
    ds_id = ds.generate(con, name, n, strategy=strategy, seed=seed, description=description)
    console.print(f"[green]Created draft dataset[/] [cyan]{ds_id}[/] (status=draft)")
    console.print("Review with: oasis-llm dataset show " + ds_id)
    console.print("Approve with: oasis-llm dataset approve " + ds_id)


@dataset_app.command("show")
def dataset_show(dataset_id: str):
    """Show details + image list for a dataset."""
    from . import datasets as ds
    con = connect()
    d = ds.get(con, dataset_id)
    if d is None:
        console.print(f"[red]No dataset {dataset_id}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{d.dataset_id}[/]  name={d.name}  status={d.status}  source={d.source}")
    if d.description:
        console.print(f"  description: {d.description}")
    if d.generation_params:
        console.print(f"  generation: {json.dumps(d.generation_params)}")
    console.print(f"  created_at={d.created_at}  approved_at={d.approved_at}")
    console.print(f"  active/total: {d.active_count}/{d.image_count}")
    rows = ds.images(con, dataset_id)
    t = Table(title=f"Images in {dataset_id}")
    for col in ("image_id", "excluded", "note"):
        t.add_column(col)
    for r in rows:
        t.add_row(r["image_id"], "yes" if r["excluded"] else "", r["note"] or "")
    console.print(t)


@dataset_app.command("approve")
def dataset_approve(dataset_id: str):
    """Lock a dataset as approved (immutable)."""
    from . import datasets as ds
    con = connect()
    try:
        ds.approve(con, dataset_id)
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Approved[/] {dataset_id}")


@dataset_app.command("archive")
def dataset_archive(dataset_id: str):
    """Mark a dataset as archived."""
    from . import datasets as ds
    con = connect()
    try:
        ds.archive(con, dataset_id)
    except (KeyError, PermissionError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[yellow]Archived[/] {dataset_id}")


@dataset_app.command("exclude")
def dataset_exclude(
    dataset_id: str,
    image_id: str,
    note: str | None = typer.Option(None, help="Reason for exclusion."),
):
    """Mark an image as excluded in a draft dataset."""
    from . import datasets as ds
    con = connect()
    try:
        ds.set_excluded(con, dataset_id, image_id, True, note=note)
    except (KeyError, PermissionError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[yellow]Excluded[/] {image_id} from {dataset_id}")


@dataset_app.command("include")
def dataset_include(dataset_id: str, image_id: str):
    """Re-include a previously excluded image."""
    from . import datasets as ds
    con = connect()
    try:
        ds.set_excluded(con, dataset_id, image_id, False)
    except (KeyError, PermissionError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Included[/] {image_id} in {dataset_id}")


@dataset_app.command("add")
def dataset_add(dataset_id: str, image_id: str):
    """Add a new image to a draft dataset."""
    from . import datasets as ds
    con = connect()
    try:
        ds.add_image(con, dataset_id, image_id)
    except (KeyError, ValueError, PermissionError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Added[/] {image_id} to {dataset_id}")


@dataset_app.command("duplicate")
def dataset_duplicate(dataset_id: str, new_name: str):
    """Clone an existing dataset's active images into a new draft."""
    from . import datasets as ds
    con = connect()
    new_id = ds.duplicate(con, dataset_id, new_name)
    console.print(f"[green]Cloned[/] {dataset_id} → [cyan]{new_id}[/] (draft)")


@dataset_app.command("delete")
def dataset_delete(
    dataset_id: str,
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
):
    """Delete a dataset (built-ins cannot be deleted)."""
    from . import datasets as ds
    con = connect()
    if not yes:
        typer.confirm(f"Delete dataset {dataset_id}?", abort=True)
    try:
        ds.delete(con, dataset_id)
    except (KeyError, PermissionError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[red]Deleted[/] {dataset_id}")


# ---------------------------------------------------------------------------
# Experiment subcommands
# ---------------------------------------------------------------------------

@experiment_app.command("list")
def experiment_list():
    """List all experiments."""
    from . import experiments as ex
    con = connect()
    rows = ex.list_all(con)
    t = Table(title="Experiments")
    for col in ("id", "name", "dataset", "status", "configs", "created"):
        t.add_column(col)
    for e in rows:
        t.add_row(
            e.experiment_id, e.name, e.dataset_id, e.status,
            str(len(e.configs)),
            str(e.created_at) if e.created_at else "-",
        )
    console.print(t)


@experiment_app.command("create")
def experiment_create(
    yaml_path: Path = typer.Argument(..., help="Path to experiment YAML."),
):
    """Create an experiment from a YAML file."""
    from . import experiments as ex
    name, dataset_id, configs, desc = ex.from_yaml(yaml_path)
    con = connect()
    try:
        exp_id = ex.create(con, name, dataset_id, configs, description=desc)
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Created[/] experiment [cyan]{exp_id}[/] with {len(configs)} configs")
    console.print(f"Run with: oasis-llm experiment run {exp_id}")


@experiment_app.command("show")
def experiment_show(experiment_id: str):
    """Show experiment details + per-config progress."""
    from . import experiments as ex
    con = connect()
    e = ex.get(con, experiment_id)
    if e is None:
        console.print(f"[red]No experiment {experiment_id}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{e.experiment_id}[/]  name={e.name}  status={e.status}")
    if e.description:
        console.print(f"  description: {e.description}")
    console.print(f"  dataset: {e.dataset_id}")
    console.print(f"  created_at={e.created_at}  finished_at={e.finished_at}")
    prog = ex.progress(con, experiment_id)
    t = Table(title="Configs")
    for col in ("config", "model", "samples", "done/total", "failed", "cost", "avg_ms"):
        t.add_column(col)
    by_name = {c.config_name: c.config_json for c in e.configs}
    for p in prog:
        cfg = by_name.get(p["config_name"], {})
        t.add_row(
            p["config_name"],
            cfg.get("model", "-"),
            str(cfg.get("samples_per_image", "-")),
            f"{p['done']}/{p['total']}",
            str(p["failed"]),
            f"${p['cost_usd']:.4f}",
            str(p["avg_latency_ms"] or "-"),
        )
    console.print(t)


@experiment_app.command("run")
def experiment_run(experiment_id: str):
    """Execute every config in the experiment sequentially."""
    from . import experiments as ex
    con = connect()
    e = ex.get(con, experiment_id)
    if e is None:
        console.print(f"[red]No experiment {experiment_id}[/]")
        raise typer.Exit(1)
    ex.update_status(con, experiment_id, "running")
    for c in e.configs:
        # Rebuild RunConfig from stored payload
        payload = dict(c.config_json)
        payload["name"] = c.run_id
        payload["image_set"] = e.dataset_id
        cfg = RunConfig(**payload)
        pending = con.execute(
            "SELECT count(*) FROM trials WHERE run_id=? AND status IN ('pending','failed')",
            [c.run_id],
        ).fetchone()[0]
        console.print(
            f"[bold cyan]>>> {c.config_name}[/] ({c.run_id}) — {pending} pending"
        )
        if pending > 0:
            asyncio.run(run_async(cfg, con))
    ex.update_status(con, experiment_id, "done")
    console.print(f"[green]Experiment {experiment_id} complete.[/]")


@experiment_app.command("delete")
def experiment_delete(
    experiment_id: str,
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
):
    """Delete an experiment and its associated runs/trials."""
    from . import experiments as ex
    if not yes:
        typer.confirm(f"Delete experiment {experiment_id} and ALL its trials?", abort=True)
    con = connect()
    ex.delete(con, experiment_id)
    console.print(f"[red]Deleted[/] {experiment_id}")


if __name__ == "__main__":
    app()
