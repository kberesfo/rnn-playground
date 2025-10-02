# RNN Playground


## Scripts
```
pipx install hatch        # or: pip install --user hatch
hatch env create          # creates the default env and installs .[dev]
hatch run test            # runs pytest
hatch run lint            # ruff check
hatch run fmt             # ruff format
hatch run typecheck       # mypy
hatch run lab             # register a Jupyter kernel for this env
```

```
import comet_ml
# TODO: ENTER YOUR API KEY HERE!! instructions above
COMET_API_KEY = os.environ['COMET_API_KEY']

assert torch.cuda.is_available(), "Please enable GPU from runtime settings"
assert COMET_API_KEY != "", "Please insert your Comet API Key"
```