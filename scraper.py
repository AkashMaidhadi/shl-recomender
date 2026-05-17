import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"

def scrape_catalog():
    headers = {"User-Agent": "Mozilla/5.0"}
    assessments = []

    # SHL catalog uses pagination with ?start=0&type=1 for Individual Tests
    start = 0
    while True:
        url = f"{BASE_URL}?start={start}&type=1"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        # Find assessment rows in the table
        rows = soup.select("tr[data-course-id]")  # adjust selector if needed
        if not rows:
            break

        for row in rows:
            name_tag = row.select_one("td.custom__table-heading__title a")
            if not name_tag:
                continue

            name = name_tag.text.strip()
            relative_url = name_tag.get("href", "")
            full_url = f"https://www.shl.com{relative_url}"

            # Test type badges (K=Knowledge, P=Personality, etc.)
            type_tags = row.select("td span.product-catalogue__key")
            test_types = [t.text.strip() for t in type_tags]

            # Remote/adaptive flags
            remote = bool(row.select_one("td.catalogue__circle.-yes"))

            assessments.append({
                "name": name,
                "url": full_url,
                "test_types": test_types,
                "remote_testing": remote,
                "description": ""  # we'll enrich this next
            })

        start += 12  # SHL paginates by 12
        time.sleep(1)  # be polite

    # Optionally enrich each with description from detail page
    for item in assessments:
        try:
            detail = requests.get(item["url"], headers=headers)
            dsoup = BeautifulSoup(detail.text, "html.parser")
            desc_tag = dsoup.select_one("div.product-catalogue__description")
            if desc_tag:
                item["description"] = desc_tag.text.strip()
            time.sleep(0.5)
        except:
            pass

    with open("catalog.json", "w") as f:
        json.dump(assessments, f, indent=2)

    print(f"Scraped {len(assessments)} assessments.")

if __name__ == "__main__":
    scrape_catalog()