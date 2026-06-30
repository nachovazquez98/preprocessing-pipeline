# Contributing

Thanks for contributing to `preprocessing_pipeline`.

This project is a small, reusable preprocessing library, so the main contribution goal is clarity. New code should make the pipeline easier to trust, easier to inspect, and easier to reuse.

## Development Setup

This guide assumes your terminal is already inside `preprocessing_pipeline/`.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the package in editable mode:

```bash
pip install -e .
```

Optional quick environment check:

```bash
python -c "import preprocessing_pipeline, pandas, numpy, sklearn; print('dev environment ready')"
```

## Project Layout

If you are new to the codebase, read `CODEBASE_GUIDE.md` first.

Short version:

- `library.py` contains the main pipeline class
- `stages.py` contains stage logic
- `metrics.py` contains reusable math helpers
- `config.py` contains normalized config behavior
- `artifacts.py` contains shared pipeline state
- `runner.py` contains the CLI entrypoint
- `demo/` contains the runnable example

## Recommended Workflow

1. create a branch for your change
2. make the smallest coherent change you can
3. run the demo and any relevant manual checks
4. update docs if behavior, config, or outputs changed
5. open a pull request with a clear summary

## What Good Contributions Look Like

Good changes usually have one or more of these qualities:

- clearer stage behavior
- better input validation
- more readable reporting
- safer defaults
- more helpful documentation
- easier extension points for future contributors

## Code Style

Prefer code that is:

- explicit rather than clever
- easy to inspect stage by stage
- small in scope
- consistent with the existing module layout

Practical guidance:

- keep orchestration in `library.py`
- keep stage-specific behavior in `stages.py`
- keep reusable math in `metrics.py`
- keep config normalization in `config.py`
- avoid pushing unrelated responsibilities into one module

## When You Change A Stage

If you modify a pipeline stage, check all of the following:

1. the stage still reads the artifacts it expects
2. the stage still writes the artifacts later stages need
3. partial runs still behave sensibly
4. the README and stage docs still match the behavior
5. the demo still runs

## When You Add A Stage

You will usually need to update:

1. `config.py`
   Add the stage name to `DEFAULT_STAGE_ORDER`.

2. `stages.py`
   Add the new implementation and stage description.

3. `library.py`
   Register the handler in `self.stage_handlers`.

4. docs
   Update `README.md` and `STAGES.md`.

## Manual Validation

There is not a full automated test suite yet, so manual validation matters.

Before opening a PR, run these checks when relevant.

Run the demo:

```bash
python demo/run_demo.py
```

Run the CLI path:

```bash
python runner.py --config demo/demo_config.py --reports-dir demo/demo_reports
```

Run a partial pipeline example:

```bash
python runner.py --config demo/demo_config.py --start utils --end bivariate
```

Verify package import:

```bash
python -c "from preprocessing_pipeline import PreprocessingPipeline; print(PreprocessingPipeline)"
```

## Documentation Expectations

Update documentation when you change:

- config fields
- stage names
- output artifacts
- report behavior
- install steps
- demo commands

Main docs to keep aligned:

- `README.md`
- `STAGES.md`
- `NOTEBOOK_USAGE.md`
- `demo/README.md`
- `CODEBASE_GUIDE.md`

## Pull Request Checklist

Before submitting, confirm:

- the change has a clear purpose
- the demo still runs
- the CLI still runs
- docs match the new behavior
- no generated noise was added accidentally

Examples of generated noise to avoid committing:

- `.venv/`
- `__pycache__/`
- generated demo reports
- temporary notebooks or scratch files

## Issues And Improvement Areas

Especially useful contribution areas include:

- automated tests
- more robust config validation
- cleaner report formatting
- richer demo coverage
- clearer error messages
- support for additional open file formats

## Questions

If you are unsure where a change belongs, start from `CODEBASE_GUIDE.md` and follow the module boundaries there. In most cases, small focused changes are easier to review and safer to maintain than broad refactors.
