# PYTHON stage
FROM python:3.12-slim AS python-base

ENV APP_DIR="/app" \
    PYSETUP_PATH="/opt/pysetup" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIPX_HOME="/opt/pipx" \
    PIPX_BIN_DIR="/opt/pipx/bin"

ENV PATH="${PIPX_BIN_DIR}:$PATH"


# BUILDER stage
FROM python-base AS builder-base
RUN apt update -y

RUN python3 -m pip install pipx
RUN pipx install poetry

WORKDIR $PYSETUP_PATH
COPY pyproject.toml poetry.lock ./

# cache packages
RUN poetry install --no-root --no-interaction


# DEVELOPMENT stage
FROM python-base AS dev
ENV CURRENT_BUILD="dev"

# copy pipx for poetry runtime
COPY --from=builder-base $PIPX_BIN_DIR $PIPX_BIN_DIR
COPY --from=builder-base $PIPX_HOME $PIPX_HOME 

WORKDIR $APP_DIR

# cached packages
COPY pyproject.toml poetry.lock README.md ./
COPY --from=builder-base $PYSETUP_PATH/.venv ./.venv

# main app code
COPY test/ ./test/
COPY unittesting/ ./unittesting/

RUN poetry install --no-interaction

ENTRYPOINT [ "poetry", "run"]
CMD [ "python3", "test/model_testing.py" ]