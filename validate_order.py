"""Независимая проверка результата модели."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(result: dict) -> list[str]:
    required = ["seo_title", "seo_description", "key_characteristics", "search_queries", "commercial_block", "missing_data", "assumptions", "questions"]
    errors = [f"Нет поля {field}" for field in required if field not in result]
    if "seo_title" in result and len(result["seo_title"]) > 70:
        errors.append("SEO заголовок длиннее 70 символов")
    if "seo_description" in result and not 800 <= len(result["seo_description"]) <= 1200:
        errors.append("SEO описание должно быть от 800 до 1200 символов")
    if "search_queries" in result and not 3 <= len(result["search_queries"]) <= 5:
        errors.append("Поисковых запросов должно быть от 3 до 5")
    for field in ("key_characteristics", "search_queries", "missing_data", "assumptions", "questions"):
        if field in result and not isinstance(result[field], list):
            errors.append(f"Поле {field} должно быть массивом")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python validate_order.py result.json")
        raise SystemExit(2)
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if isinstance(value, list):
        errors = [error for item in value for error in validate(item.get("draft", item))]
    else:
        errors = validate(value)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Результат прошёл проверку")
