"""AI помощник для подготовки B2B описаний измерительных приборов."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()


class Product(BaseModel):
    name: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @field_validator("specs")
    @classmethod
    def clean_specs(cls, value: dict[str, str]) -> dict[str, str]:
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}


class ProductDraft(BaseModel):
    seo_title: str
    seo_description: str
    key_characteristics: list[str]
    search_queries: list[str]
    commercial_block: str
    missing_data: list[str]
    assumptions: list[str]
    questions: list[str]


class GenerateRequest(Product):
    demo: bool = False


SYSTEM_PROMPT = """Ты редактор B2B интернет-магазина контрольно-измерительных приборов.
Создай черновик карточки товара только по данным из входного JSON.

Критические правила.
1. Нельзя придумывать технические характеристики, диапазоны, классы точности, материалы, интерфейсы, сертификаты, размеры, назначение или совместимость.
2. Любые числа и единицы измерения можно использовать только если они есть во входных данных.
3. Если данных для утверждения нет, напиши «данных нет» или добавь поле в missing_data и вопрос в questions.
4. Не называй товар лучшим, точным, надёжным или сертифицированным без основания во входных данных.
5. SEO текст должен быть полезным инженеру и закупщику, без рекламных обещаний.
6. Верни только JSON по указанной схеме, без Markdown вокруг JSON.

Порядок ответа.
seo_title до 70 символов.
seo_description от 800 до 1200 символов.
key_characteristics это список строк вида «Название: значение» только по входным specs.
search_queries от 3 до 5 запросов на основе названия, производителя и указанных характеристик.
commercial_block это короткий текст о том, кому прибор подходит и какую задачу решает. Используй только назначение из description и характеристики из входа.
missing_data это действительно отсутствующие данные, которые важны для публикации.
assumptions и questions нужны для неоднозначных или неполных мест.
"""


def user_prompt(product: Product) -> str:
    return "Входные данные товара:\n" + json.dumps(product.model_dump(), ensure_ascii=False, indent=2)


def demo_description(product: Product) -> str:
    specs = "; ".join(f"{k}: {v}" for k, v in product.specs.items())
    base = (
        f"{product.name} производства {product.manufacturer} предназначен для задач, описанных в технических данных товара. "
        f"При выборе и заказе прибора следует ориентироваться на указанные характеристики и условия применения из исходного описания. "
        f"Ключевые параметры модели: {specs}. "
        f"{product.description} "
        "Карточка подготовлена как технический черновик для B2B интернет магазина. Перед публикацией специалисту следует сверить обозначение модели, "
        "условия эксплуатации, способ установки и комплект поставки с паспортом или официальной документацией производителя. "
        "Если нужного параметра нет во входной заявке, он не подменяется предположением. Это позволяет использовать описание для предварительного подбора "
        "и одновременно избежать появления неподтверждённых характеристик в каталоге. "
    )
    while len(base) < 800:
        base += " Окончательные условия поставки и применения необходимо подтвердить по технической документации."
    return base[:1200]


def demo_draft(product: Product) -> ProductDraft:
    specs = [f"{k}: {v}" for k, v in product.specs.items()]
    queries = [product.name, f"{product.manufacturer} {product.name}"]
    for key in list(product.specs)[:3]:
        queries.append(f"{product.name} {product.specs[key]}")
    missing = []
    for field in ("материал корпуса", "класс защиты", "межповерочный интервал", "комплект поставки"):
        if field not in product.specs:
            missing.append(field)
    return ProductDraft(
        seo_title=f"{product.name} {product.manufacturer}"[:70],
        seo_description=demo_description(product),
        key_characteristics=specs,
        search_queries=list(dict.fromkeys(queries))[:5],
        commercial_block=f"Прибор подходит для задач, описанных в исходном техническом описании. Перед заказом нужно сопоставить диапазон измерений, присоединение и остальные параметры с условиями эксплуатации.",
        missing_data=missing,
        assumptions=[],
        questions=["Уточнить параметры, перечисленные в missing_data, если они обязательны для выбора."],
    )


def check_draft(product: Product, draft: ProductDraft) -> list[str]:
    errors: list[str] = []
    if len(draft.seo_title) > 70:
        errors.append("SEO заголовок длиннее 70 символов")
    if not 800 <= len(draft.seo_description) <= 1200:
        errors.append("SEO описание должно быть длиной от 800 до 1200 символов")
    source_values = set(product.specs.values())
    for characteristic in draft.key_characteristics:
        if ":" in characteristic:
            value = characteristic.split(":", 1)[1].strip()
            if value and value not in source_values and value.lower() != "данных нет":
                errors.append(f"Характеристика не найдена во входных данных: {value}")
    if not 3 <= len(draft.search_queries) <= 5:
        errors.append("Поисковых запросов должно быть от 3 до 5")
    return errors


def llm_draft(product: Product) -> ProductDraft:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан. Для локальной проверки используйте demo=true")
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(product)}],
    )
    content = response.choices[0].message.content or "{}"
    return ProductDraft.model_validate(json.loads(content))


def generate(product: Product, demo: bool = False) -> dict[str, Any]:
    draft = demo_draft(product) if demo else llm_draft(product)
    errors = check_draft(product, draft)
    return {"product": product.model_dump(), "draft": draft.model_dump(), "validation_errors": errors}


app = FastAPI(title="AI Product Copywriter", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate_endpoint(request: GenerateRequest) -> dict[str, Any]:
    try:
        return generate(Product.model_validate(request.model_dump(exclude={"demo"})), request.demo)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация черновика карточки измерительного прибора")
    parser.add_argument("--input", default="examples/products.json")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--demo", action="store_true", help="не использовать API, показать локальный пример")
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        source = [source]
    result = [generate(Product.model_validate(item), args.demo) for item in source]
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово. Обработано товаров: {len(result)}. Результат: {args.output}")


if __name__ == "__main__":
    main()
