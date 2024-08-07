"""
directed-networks scrapping from https://networks.skewed.de/
"""
import os
import requests
from bs4 import BeautifulSoup
import json
import zipfile

#PARAMS 

# max scraping limit
max_networks = 20
# Base URL of the website
base_url = "https://networks.skewed.de"
# Directory to save the downloaded files
save_dir = "directed_networks"
directed_networks = {}
def download_file(url, dataset_name, save_subdir= ""):
    print(f"fetching file from {url}")
    response = requests.get(url)
    #if not succesful return false,
    if response.status_code != 200:
        print(f"unable to fetch file from {url}")
        return False
    else:
        filename = f"{save_subdir}/{dataset_name.split('.')[0]}/{dataset_name}"
        #if doesn't exist, create the directory
        if not os.path.exists(f"{save_subdir}/{dataset_name.split('.')[0]}"):
            os.makedirs(f"{save_subdir}/{dataset_name.split('.')[0]}")
        print(f"filename: {filename}")
        with open(filename, 'wb') as file:
            file.write(response.content)
        #if zip file, extract it
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(f"{save_subdir}/{dataset_name.split('.')[0]}")
                
        print(f"Downloaded: {filename}")
        return True
        

if __name__ == "__main__":
    # Create the directory if it doesn't exist
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all('tr')
    for row in rows:

        columns = row.find_all('td')
        if len(columns) > 0:
            kind = columns[12].get_text(strip=True)
            
            #print name of network and kind
            name = columns[0].get_text(strip=True)
            print(f"Name: {name}")
            print(f"Kind: {kind}")
            if kind.lower() == 'directed':
                network_link = columns[0].find('a')['href']
                network_url = base_url + network_link
                
                # Fetch the network page
                print(f"Fetching: {network_url}")
                network_response = requests.get(network_url)
                network_soup = BeautifulSoup(network_response.content, 'html.parser')
                
                # Find the download link for the network file
                download_links = network_soup.find_all('a', href=True)
                # print(f"download_links: {download_links}")
                for link in download_links:
                    #download only csv version
                    if 'csv' in link['href']:
                        
                        download_url = base_url + "/net/" + link['href']
                        #filees have https://networks.skewed.de/dom/files/SatohOhkawara_2008a.csv.zip format, extract the name 
                        #between files/ and .csv.zip and append it to the current 
                        dataset_name = download_url.split('/')[-1]
                        #make dir if save_dir + f"/{name}" does not exist
                        save_subdir = save_dir + f"/{name}"
                        if not os.path.exists(save_subdir):
                            os.makedirs(save_subdir)
                        
                        if download_file(download_url, dataset_name,  save_subdir= save_subdir):
                            directed_networks[f"{name}_{dataset_name}"] = download_url
                            # print(f"directed_networks: {directed_networks}")
                        
                        if len(directed_networks) >= max_networks:
                            break
                        
        if len(directed_networks) >= max_networks:
            break
        #

                    
    #save the dictionary to a json file
    with open('directed_networks.json', 'w') as f:
        json.dump(directed_networks, f, indent=4)
    print("Saved directed networks to directed_networks.json")