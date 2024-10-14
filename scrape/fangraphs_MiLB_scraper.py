import time
import os
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException


def initialize_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enable DevTools for Chrome
    
    # Get ChromeDriver path from environment variable
    chrome_driver_path = os.getenv("CHROMEDRIVER_PATH")
    if not chrome_driver_path:
        raise EnvironmentError("Please set the CHROMEDRIVER_PATH environment variable")
    
    service = Service(chrome_driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)


def download_table_html(driver, url, save_path):
    driver.get(url)

    try:
        # Wait for the page controls to load
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.page-item-control")))

        # Select "Infinity" from the page control select element
        page_control_div = driver.find_element(By.CSS_SELECTOR, "div.page-item-control")
        page_control_select = page_control_div.find_element(By.TAG_NAME, "select")
        select = Select(page_control_select)
        select.select_by_value("2000000000")

        # Wait for the table to load
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-fixed")))

        # Now find the table within the table-fixed div
        table_fixed = driver.find_element(By.CSS_SELECTOR, "div.table-fixed")
        table_html = table_fixed.get_attribute('outerHTML')

        # Save the table HTML to a file
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(table_html)
        print(f"Table for {url} saved to {save_path}")

    except TimeoutException:
        print(f"Timeout occurred when loading page for {url}.")
    except Exception as e:
        print(f"Error downloading table for {url}: {e}")


# Main function to download tables for each year
def main():
    driver = initialize_driver()
    years = list(range(2017, 2025))  # 2017 to 2024
    base_save_dir = "data/tables/"

    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    for year in years:
        # Construct the URL for each year
        # Note: The URL for 2019-2024 is different from 2017-2018
        # get the summary url's, then the scouting url's
        save_path = os.path.join(base_save_dir, f"prospects_{year}_summary.html")
        download_table_html(driver, f"https://www.fangraphs.com/prospects/the-board/{year}-prospect-list/summary", save_path)

        if year in [2017, 2018]:
            url = f"https://www.fangraphs.com/prospects/the-board/{year}-prospect-list/scouting"
            save_path = os.path.join(base_save_dir, f"prospects_{year}.html")
            download_table_html(driver, url, save_path)
        else:
            # get the pitcher and hitter scouting reports separately
            hit_url = f"https://www.fangraphs.com/prospects/the-board/{year}-prospect-list/scouting-position"
            pit_url = f"https://www.fangraphs.com/prospects/the-board/{year}-prospect-list/scouting-pitching"
            save_path = os.path.join(base_save_dir, f"prospects_{year}_hitters.html")
            download_table_html(driver, hit_url, save_path)
            save_path = os.path.join(base_save_dir, f"prospects_{year}_pitchers.html")
            download_table_html(driver, pit_url, save_path)

    driver.quit()
    print("All tables downloaded!")


if __name__ == "__main__":
    main()