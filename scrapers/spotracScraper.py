import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Initialize the Selenium WebDriver (headless mode for efficiency)
def initialize_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    
    # Get ChromeDriver path from environment variable
    chrome_driver_path = os.getenv("CHROMEDRIVER_PATH")
    if not chrome_driver_path:
        raise EnvironmentError("Please set the CHROMEDRIVER_PATH environment variable")
    
    service = Service(chrome_driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)

# Fetch and parse player contract details using Selenium
def parse_player_contract(player_url, player_name, team_abbr, csv_writer, driver):
    driver.get(player_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-white.premium.contract-breakdown")))

    soup = BeautifulSoup(driver.page_source, "html.parser")
    contract_table = soup.find("table", {"class": "table table-white premium contract-breakdown"})
    
    if contract_table:
        rows = contract_table.find("tbody").find_all("tr")
        for row in rows:
            columns = row.find_all("td")
            year = columns[0].text.strip() if len(columns) > 0 else None
            status = columns[3].text.strip() if len(columns) > 3 else None
            payroll = columns[4].text.strip() if len(columns) > 4 else None
            luxury_tax = columns[5].text.strip() if len(columns) > 5 else None
            
            csv_writer.writerow([player_name, team_abbr, year, status, payroll, luxury_tax])

# Extract and process player data from team page using Selenium
def extract_players_from_team(team_abbr, driver):
    team_url = f"https://www.spotrac.com/mlb/contracts/_/team/{team_abbr}/sort/value"
    driver.get(team_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody[aria-live='polite']")))

    soup = BeautifulSoup(driver.page_source, "html.parser")
    player_table = soup.find("tbody", {"aria-live": "polite"})
    
    if player_table:
        player_rows = player_table.find_all("tr")
        player_data = [
            {
                "player_url": f"{row.find('a')['href']}/{row.find('a').text.lower().replace(' ', '-')}",
                "player_name": row.find('a').text.strip()
            }
            for row in player_rows if row.find('a', href=True)
        ]
        return player_data
    return []

# Scrape player data concurrently using Selenium
def scrape_team(team_abbr, driver, csv_writer):
    try:
        print(f"Scraping team: {team_abbr}")
        player_data_list = extract_players_from_team(team_abbr, driver)
        
        for player_data in player_data_list:
            parse_player_contract(player_data["player_url"], player_data["player_name"], team_abbr, csv_writer, driver)
            
    except Exception as e:
        print(f"Error scraping {team_abbr}: {e}")

# Main function to run the scraper
def main():
    driver = initialize_driver()
    team_abbreviations = [
        "ari", "atl", "bal", "bos", "chc", "chw", "cin", "cle", "col", "det", "hou", "kc", "laa", "lad", "mia",
        "mil", "min", "nym", "nyy", "ath", "phi", "pit", "sd", "sf", "sea", "stl", "tb", "tex", "tor", "wsh"
    ]
    
    # Use a relative path for the CSV file
    csv_file = "./data/mlb_salary_data.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with open(csv_file, mode="w", newline='', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Player Name", "Team", "Year", "Status", "Payroll", "Luxury Tax"])

        for team_abbr in team_abbreviations:
            scrape_team(team_abbr, driver, csv_writer)

    driver.quit()
    print(f"Scraping complete! Data saved to {csv_file}")

if __name__ == "__main__":
    main()