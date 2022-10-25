# ChivesBeacon

### Run locally

Set up a Python 3 virtualenv and install the dependencies on it:

```bash
python3 -m venv venv
source venv/PATH/activate
pip install -r requirements.txt
flask run
```

### Package dependencies

When install new package, please run 

```bash
pip install package && pip freeze > requirements.txt
```