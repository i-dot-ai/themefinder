# Internal i.AI contributors to ThemeFinder

At present we are not accepting contributions to `themefinder` as it is undergoing rapid development within i.AI. If you have a suggestion, please raise an [issue](https://github.com/i-dot-ai/themefinder/issues) or email us at [packages@cabinetoffice.gov.uk](mailto:packages@cabinetoffice.gov.uk).


## Architecture decisions

## Developing the package locally

Ensure you have installed `pre-commit` in the repo: `pre-commit install`. This includes hooks to prevent you committing sensitive data. This will not catch everything; always take care when working in the open.

We use `poetry` to develop the `themefinder` - `poetry install` to install packages.

If you wish to install your local development version of the package into another project to test it, you will need to install in "editable" mode:
```
pip install -e <FILE_PATH>
```
or 
```
poetry add -e <FILE_PATH>
```
where `<FILE_PATH>` is the location of your local version of `themefinder`.


## Evaluation

The `evals/` directory contains our benchmarking evaluation suite used to measure system performance. When you make a change to the `themefinder` pipeline, run the evaluations to ensure you haven't reduced performance. 

The `eval_mapping` and `eval_sentiment` evaluations use sensitive data stored in our AWS environment. These specific evaluations will only function with proper AWS account access and credentials. Similarly, the `make run_evals` command assumes you have AWS access configured.

These evaluations use the Azure Open AI endpoint.

### Running the evaluations

Set your environment variables: copy `.env.example` to `.env` and populate with the name of the S3 bucket, and the details for the Azure endpoint.

Install packages for this repo: `poetry install`.

Ensure you have AWS access set up, and assume your AWS role to allow you to access the data.

These evaluations can be executed either:
- By running `make run_evals` to execute the complete evaluation suite (or `poetry run make run_evals` if you're using `poetry`)
- By directly running individual evaluation files that begin with `eval_` prefix

Note that the evals specifically use GPT-4o, and JSON structured output.


## Releasing to PyPi

Creating a GitHub release will push that version of the package to TestPyPi and then PyPi.

1. Check with the Consult engineering team in Slack that it is ok to do a release.
2. Update the version number in `pyproject.toml` - note that we are using SemVer.
3. From the `main` branch, create a [pre-release](https://github.com/i-dot-ai/themefinder/releases) by ticking the box at the bottom of the release. The release should have the tag `vX.Y.Z` where `X.Y.Z` is the version number in `pyproject.toml`.
4. Use the "Generate release notes" button to get you started on writing a suitable release note.
5. Creating the pre-release should trigger a deployment to [TestPyPi](https://test.pypi.org/project/themefinder/). Check the GitHub Actions and TestPyPi to ensure that this happens.
6. Once you're happy, go back to the pre-release and turn it into a [release](https://github.com/i-dot-ai/themefinder/releases).
7. When you publish the release, this will trigger a deployment to PyPi. You can check the GitHub actions and [PyPi](https://pypi.org/project/themefinder/) itself to confirm the deployment has worked.


## Docs

These docs are built using [MkDocs](https://www.mkdocs.org/), and are deployed to [GitHub Pages](https://i-dot-ai.github.io/themefinder/) when a pull request is merged to the `main` branch. The docs are stored in the `docs` folder as Markdown files.

To test your changes locally:
```
poetry run mkdocs build
poetry run mkdocs serve
```
and go to [http://127.0.0.1:8000/i-dot-ai/themefinder/](http://127.0.0.1:8000/i-dot-ai/themefinder/) in the browser.


## Architecture decision records (ADR)

If you are making significant changes, please record them in the architecure documents. We are using [adr-tools](https://github.com/npryce/adr-tools) - see the ADR tools repo for how to install and use.




## Langfuse integration

Langfuse can be used to monitor costs and log LLM calls. This can be initialised from outside of the ThemeFinder package when the LLM is instantiated. 

Set up:
Add to the `.env` file:
```
# Langfuse
LANGFUSE_SECRET_KEY=""
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_HOST=""
```
using the keys from you own langfuse instance.

```python
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
 
dotenv.load_dotenv() 

# Initialize Langfuse CallbackHandler for Langchain (tracing)
# Use the session id to group calls
langfuse_callback_handler = CallbackHandler(session_id="run_1")

# Initialise your LLM of choice using langchain
llm = AzureChatOpenAI(
    model="gpt-4o",
    temperature=0,
    callbacks=[langfuse_callback_handler],
    model_kwargs={"response_format": {"type": "json_object"}},
)
```

Use as you would normally. Your langfuse dashboard should log the llm calls including the inputs, outputs, model name, costs, and more.
