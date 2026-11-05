from App.services.matching import match_products


def main() -> None:
    checked = [{"id": "t1", "name": "Молоко 1.5% ультрапастеризованное, 1 л"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Молоко 2.5% пастеризованное 1 л"},
        {"id": "s3", "name": "Сметана 20% 500 г"},
    ]

    result = match_products(checked, stock, top_n=3)
    print(result)


if __name__ == "__main__":
    main()
