import json
import os
from datetime import datetime
from typing import Any, Dict, List


def append_review_item(item: Dict[str, Any], queue_path: str = "review_queue.json") -> None:
    os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)
    queue = load_review_queue(queue_path)
    item = dict(item)
    item["created_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    queue.append(item)
    with open(queue_path, "w", encoding="utf-8") as fh:
        json.dump(queue, fh, indent=2)


def load_review_queue(queue_path: str = "review_queue.json") -> List[Dict[str, Any]]:
    if not os.path.exists(queue_path):
        return []
    with open(queue_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
