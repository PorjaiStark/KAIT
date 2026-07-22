import os
import csv


class Tee:
    """Duplicates writes to multiple streams (e.g. real stdout + a log file),
    so the exact console output of a run is captured for later reading."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def append_run_log(row, path):
    """Append one summary row to a master CSV log. Columns are the union of
    keys seen so far -- if `row` introduces a field earlier rows didn't
    have, the whole file is rewritten with the wider header (old rows get
    a blank in the new column) so the header always matches every row's
    column count. Cheap in practice: these logs are one row per training/
    eval run, never a hot path."""

    file_exists = os.path.isfile(path)

    if not file_exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    new_fields = [k for k in row.keys() if k not in existing_fields]

    if new_fields:
        fieldnames = existing_fields + new_fields
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
    else:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow(row)
