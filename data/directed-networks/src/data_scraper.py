"""
directed-networks scrapping from https://networks.skewed.de/
"""
import os
import requests
from bs4 import BeautifulSoup
import json
import zipfile



data_sets_validation = {
    "Political blogs": ["polblogs"],
    "Tadpole larva brain (C. intestinalis)": ["cintestinalis"],
    "Little Rock Lake food web": ["foodweb_little_rock"],
    "FAO trade network": ["fao_trade"],
    "Advogato trust network": ["advogato"],
    "Manufacturing company email": ["email_company"],
    "Primary school contacts": [
        "dutch_school_klas12b_primary",
        "dutch_school_klas12b_net_1",
        "dutch_school_klas12b_net_2",
        "dutch_school_klas12b_net_3",
        "dutch_school_klas12b_net_4"
    ],
    "U.S. government agency websites": [
        "us_agencies_alabama",
        "us_agencies_alaska",
        "us_agencies_arizona",
        "us_agencies_arkansas",
        "us_agencies_california",
        "us_agencies_colorado",
        "us_agencies_connecticut",
        "us_agencies_delaware",
        "us_agencies_florida",
        "us_agencies_georgia",
        "us_agencies_hawaii",
        "us_agencies_idaho",
        "us_agencies_illinois",
        "us_agencies_indiana",
        "us_agencies_iowa",
        "us_agencies_kansas",
        "us_agencies_kentucky",
        "us_agencies_louisiana",
        "us_agencies_maine",
        "us_agencies_maryland",
        "us_agencies_massachusetts",
        "us_agencies_michigan",
        "us_agencies_minnesota",
        "us_agencies_mississippi",
        "us_agencies_missouri",
        "us_agencies_montana",
        "us_agencies_nebraska",
        "us_agencies_nevada",
        "us_agencies_newhampshire",
        "us_agencies_newjersey",
        "us_agencies_newmexico",
        "us_agencies_newyork",
        "us_agencies_northcarolina",
        "us_agencies_northdakota",
        "us_agencies_ohio",
        "us_agencies_oklahoma",
        "us_agencies_oregon",
        "us_agencies_pennsylvania",
        "us_agencies_rhodeisland",
        "us_agencies_southcarolina",
        "us_agencies_southdakota",
        "us_agencies_tennessee",
        "us_agencies_texas",
        "us_agencies_utah",
        "us_agencies_vermont",
        "us_agencies_virginia",
        "us_agencies_washington",
        "us_agencies_westvirginia",
        "us_agencies_wisconsin",
        "us_agencies_wyoming"
    ],
    "Friendship network": ["7th_graders", "add_health_comm28", "add_health_comm41", "add_health_comm62", "add_health_comm75"],
    "C. elegans connectome": ["celegansneural"],
    "Email network (Uni. R-V, Spain)": ["uni_email"],
    "Friendship network 2": ["add_health_comm15", "add_health_comm33", "add_health_comm50", "add_health_comm68", "add_health_comm79"],
    "Messel Shale food web": ["messal_shale"],
    "E. coli transcription network": ["ecoli_transcription_v1_0", "ecoli_transcription_v1_1"],
    "Copenhagen networks study (calls)": ["copenhagen_calls"],
    "U.S. government agency websites (VT)": ["us_agencies_vermont"],
    "Physician trust network": ["physician_trust"],
    "UN migration stock": ["un_migrations"],
    "Yeast transcription network": ["yeast_transcription"],
    "FAA preferred routes": ["faa_routes"],
    "Copenhagen networks study (sms)": ["copenhagen_sms"],
    "Figeys human interactome": ["interactome_figeys"],
    "Openflights airport network": ["openflights"],
    "C. elegans neurons (male, chemical)": ["celegans_2019_male_chemical_synapse", "celegans_2019_male_chemical_corrected"],
    "Chess matches": ["chess"]
}

# combine all data sets for validation into one list

dataset_allard = [ dataset for datasets in data_sets_validation.values() for dataset in datasets]

print(f"dataset_allard: {dataset_allard}")


# max scraping limit
max_networks = 50
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
            
            if kind.lower() == 'directed':
                print(f"Name: {name}, Kind: {kind}")
                if name not in dataset_allard:
                    continue
                print(f"dataset found in allard: {name}")
                
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