FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /uvx /bin/
WORKDIR /app
COPY examples/simple_client/requirements.txt /tmp/requirements.txt
COPY packages/openpi-client /tmp/openpi-client
RUN uv pip install --system -r /tmp/requirements.txt /tmp/openpi-client
COPY examples/simple_client/main.py /app/main.py
CMD ["/bin/bash", "-c", "python /app/main.py $CLIENT_ARGS"]
