FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY meetings_to_tasks ./meetings_to_tasks
RUN pip install --no-cache-dir ".[asr]"
ENTRYPOINT ["m2t", "run"]
