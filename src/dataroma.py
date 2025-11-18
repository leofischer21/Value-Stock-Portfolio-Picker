import requests
from bs4 import BeautifulSoup

def get_superinvestor_data():
    url = "https://www.dataroma.com/m/holdings.php?m=hold"
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'lxml')

    data = {}
    for row in soup.select('table tr')[1:]:
        cols = row.find_all('td')
        if len(cols) < 6: continue
        ticker = cols[0].text.strip().split()[0]
        holders = int(cols[3].text.strip()) if cols[3].text.strip().isdigit() else 0
        new_buys = int(cols[4].text.strip()) if cols[4].text.strip().isdigit() else 0
        if holders > 0:
            score = min(holders / 20, 1.0) * 0.7 + min(new_buys / 5, 1.0) * 0.3
            data[ticker] = round(score, 3)
    return data