FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "tech_challenge_fase2.api:app", "--host", "0.0.0.0", "--port", "8000"]

