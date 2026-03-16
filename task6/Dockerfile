FROM python:3.11-slim

WORKDIR /app

# cron + minimal deps
RUN apt-get update && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Script
COPY update_index.py /app/update_index.py
RUN chmod +x /app/update_index.py

# Logs dir
RUN mkdir -p /var/log

# Cron schedule: every hour by default
# You can override by mounting a different cron file if needed.
COPY crontab /etc/cron.d/indexer-cron
RUN chmod 0644 /etc/cron.d/indexer-cron && crontab /etc/cron.d/indexer-cron

# Start cron in foreground
CMD ["cron", "-f"]
