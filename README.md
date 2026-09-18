# DataInvariantBench

Code accompanying **Semantic Contracts in Text-to-SQL: A Controlled Study of
Representation and Information**, by **Shitiz Kumar, Independent Researcher**.

This repository currently contains the frozen synthetic benchmark generators,
prompt renderer, SQL execution/evaluation harness, and optional model-runner code
under `src/datainvariant/`.

## Local use

```sh
python3 -m pip install -e .
python3 -m datainvariant --help
```

The local generators and oracle checks do not require a model API. Live model
runner commands require your own credentials and can incur charges; do not run
them unless you deliberately authorize new inference and set a budget.

## Release status

The complete analysis code, paper, and privacy-redacted experimental archive have
been prepared and checked locally. Publication of the saved model responses and
AI audit records is awaiting the author's explicit data-release approval. Those
research artifacts are **not yet available from this repository**.

The manuscript is an audited descriptive study, not a completed confirmatory
study or peer-reviewed publication. Human validation remains incomplete. Do not
interpret AI verification as human review or claim arXiv acceptance.

## License and attribution

Code retains the [MIT license](LICENSE). Use [CITATION.cff](CITATION.cff) for
repository attribution. Model weights and credentials are not distributed.
