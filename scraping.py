from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from datetime import datetime, timedelta

# Fungsi untuk membagi tanggal menjadi rentang 1 hari
def generate_date_ranges(start_date, end_date, interval_days=1):
    date_ranges = []
    current_start_date = start_date
    while current_start_date < end_date:
        current_end_date = min(current_start_date + timedelta(days=interval_days - 1), end_date)
        date_ranges.append((current_start_date, current_end_date))
        current_start_date = current_end_date + timedelta(days=1)
    return date_ranges

# Setup untuk menggunakan Edge WebDriver
driver = webdriver.Edge()  # Ganti path_to_msedgedriver dengan path aktual ke Edge WebDriver

# Buka halaman Google dalam bahasa Inggris dengan parameter hl=en
driver.get("https://www.google.com/?hl=en")

# Cari input box dan masukkan kata kunci
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("jokowi")
search_box.send_keys(Keys.RETURN)  # Simulasi menekan Enter

# Klik tab "News" berdasarkan href yang berisi "tbm=nws"
news_tab = driver.find_element(By.CSS_SELECTOR, 'a[href*="tbm=nws"]')
news_tab.click()

# Klik tombol "Tools"
tools_button = driver.find_element(By.XPATH, '//div[text()="Tools"]')
tools_button.click()

# Klik span untuk mengubah urutan berita
sort_span = driver.find_element(By.XPATH, '//div[@class="AozSsc"]//span')
sort_span.click()

# Data berita
news_data = []

# Rentang waktu awal dan akhir
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 10, 5)

# Menghasilkan rentang waktu 1 hari
date_ranges = generate_date_ranges(start_date, end_date)

for start, end in date_ranges:
    print(f"Processing range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    # Scroll halaman ke atas
    driver.execute_script("window.scrollTo(0, 0);")

    # Klik elemen span untuk mengubah rentang tanggal (span dengan class 'gTl8xb')
    span_element = driver.find_element(By.CLASS_NAME, "gTl8xb")
    span_element.click()

    # Klik elemen <span> dengan teks "Custom range..."
    custom_range_element = driver.find_element(By.XPATH, '//span[text()="Custom range..."]')
    custom_range_element.click()

    # Isi tanggal mulai
    start_date_input = driver.find_element(By.CLASS_NAME, "OouJcb")
    start_date_input.clear()
    start_date_input.send_keys(start.strftime("%m/%d/%Y"))

    # Isi tanggal akhir
    end_date_input = driver.find_element(By.CLASS_NAME, "rzG2be")
    end_date_input.clear()
    end_date_input.send_keys(end.strftime("%m/%d/%Y"))

    # Klik tombol "Go"
    go_button = driver.find_element(By.XPATH, '//g-button[text()="Go"]')
    go_button.click()

    # Scroll halaman untuk memastikan semua berita muncul
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Scroll ke bawah

        # Mengumpulkan data berita dari halaman saat ini
        try:
            news_elements = driver.find_elements(By.CLASS_NAME, "SoaBEf")  # Mengambil semua elemen berita
            for news_element in news_elements:
                title_element = news_element.find_element(By.CLASS_NAME, "n0jPhd")  # Mengambil judul
                source_element = news_element.find_element(By.CLASS_NAME, "MgUUmf")  # Mengambil nama media
                link_element = news_element.find_element(By.CSS_SELECTOR, 'a.WlydOe')  # Mengambil link berita

                # Menggunakan tanggal dari `start` untuk kolom waktu terbit
                news_data.append([
                    title_element.text,
                    source_element.text,
                    start.strftime("%Y-%m-%d"),
                    link_element.get_attribute('href')
                ])
        except Exception as e:
            print(f"Error occurred while collecting news data: {e}")

        # Cek jika ada tombol untuk halaman berikutnya
        try:
            next_page_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'pnnext'))  # Tombol Next berdasarkan ID
            )
            next_page_button.click()  # Klik tombol Next
        except Exception as e:
            print(f"No more pages or error occurred: {e}")
            break  # Jika tidak ada halaman berikutnya, keluar dari loop
    
# Menyimpan data ke dalam file CSV
with open('jokowi 2019-2024.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Menulis header
    csv_writer.writerow(['Judul Berita', 'Nama Media', 'Waktu Terbit', 'Link Berita'])
    # Menulis data berita
    csv_writer.writerows(news_data)

# Optional: Tutup browser
driver.quit()
