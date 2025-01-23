# Contributing to ThemeFinder

If you have a bug report, feature request or feedback, please raise a [GitHub issue](https://github.com/i-dot-ai/themefinder/issues) on this repo, or contact us at `themefinder@cabinetoffice.gov.uk`.

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and security. Before contributing, you should set up pre-commit hooks in your local environment:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hooks:
```bash
pre-commit install
```

3. The following hooks will run automatically before each commit:

- **check-added-large-files**: Prevents committing files larger than 5MB
- **detect-secrets**: Scans for potential secrets, credentials, or sensitive information
- **detect-ip**: Prevents committing IP addresses (excludes 0.0.0.0 and 127.0.0.1)
- **detect-aws-account**: Detects and prevents AWS account numbers from being committed
- **ruff**: Python linter and formatter that ensures code style consistency
- **nbstripout**: Automatically strips output from Jupyter notebooks

You can run the hooks manually on all files at any time:
```bash
pre-commit run --all-files
```

If a hook fails, it will provide information about the failure and prevent the commit. Fix the issues and try committing again. Some hooks (like ruff) may automatically fix issues - in these cases, you'll need to stage the changes and retry the commit.
