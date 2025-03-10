FROM python:3.11.10-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into container.
COPY . /backend

# Install the application dependencies.
WORKDIR /backend
RUN uv sync --frozen --no-cache

# Run the application.
CMD ["/backend/.venv/bin/uvicorn", "app.main:app", "--port", "8080", "--host", "0.0.0.0"]