"""
directed-networks scrapping from https://networks.skewed.de/
"""
import os
import requests
from bs4 import BeautifulSoup
import json
import zipfile

#PARAMS 

#Allard Data set list:
# Names = [
#     "Political blogs",
#     "Tadpole larva brain (C. intestinalis)",
#     "Little Rock Lake food web",
#     "FAO trade network",
#     "Advogato trust network",
#     "Manufacturing company email",
#     "Primary school contacts",
#     "U.S. government agency websites",
#     "Friendship network",
#     "C. elegans connectome",
#     "Email network (Uni. R-V, Spain)",
#     "Friendship network 2",
#     "Messel Shale food web",
#     "E. coli transcription network",
#     "Copenhagen networks study (calls)",
#     "U.S. government agency websites (VT)",
#     "Physician trust network",
#     "UN migration stock",
#     "Yeast transcription network",
#     "FAA preferred routes",
#     "Copenhagen networks study (sms)",
#     "Figeys human interactome",
#     "Openflights airport network",
#     "C. elegans neurons (male, chemical)",
#     "Chess matches"
# ]

# dataset_names = [
#     '7th_graders',
#     'add_health_comm28',
#     'add_health_comm41',
#     'add_health_comm62',
#     'add_health_comm75',
#     'add_health_comms3',
#     'anybeat',
#     'bitcoin_trust',
#     'caida_as_20040301',
#     'caida_as_20040607',
#     'caida_as_20040906',
#     'caida_as_20041206',
#     'caida_as_20050307',
#     'caida_as_20050606',
#     'caida_as_20050905',
#     'caida_as_20051205',
#     'caida_as_20060116',
#     'caida_as_20060206',
#     'caida_as_20060227',
#     'caida_as_20060320',
#     'caida_as_20060410',
#     'caida_as_20060501',
#     'caida_as_20060522',
#     'caida_as_20060612',
#     'caida_as_20060703',
#     'caida_as_20060724',
#     'caida_as_20060814',
#     'caida_as_20060904',
#     'caida_as_20060925',
#     'caida_as_20061016',
#     'caida_as_20061106',
#     'caida_as_20061127',
#     'caida_as_20061218',
#     'caida_as_20070108',
#     'caida_as_20070129',
#     'caida_as_20070219',
#     'caida_as_20070312',
#     'cattle',
#     'add_health_comm15',
#     'add_health_comm33',
#     'add_health_comm50',
#     'add_health_comm68',
#     'add_health_comm79',
#     'add_health_comm8s4',
#     'bison',
#     'caida_as_20040105',
#     'caida_as_20040405',
#     'caida_as_20040705',
#     'caida_as_20041004',
#     'caida_as_20050103',
#     'caida_as_20050404',
#     'caida_as_20050704',
#     'caida_as_20051003',
#     'caida_as_20060102',
#     'caida_as_20060123',
#     'caida_as_20060213',
#     'caida_as_20060306',
#     'caida_as_20060327',
#     'caida_as_20060417',
#     'caida_as_20060508',
#     'caida_as_20060529',
#     'caida_as_20060619',
#     'caida_as_20060710',
#     'caida_as_20060731',
#     'caida_as_20060821',
#     'caida_as_20060911',
#     'caida_as_20061002',
#     'caida_as_20061023',
#     'caida_as_20061113',
#     'caida_as_20061204',
#     'caida_as_20061225',
#     'caida_as_20070115',
#     'caida_as_20070205',
#     'caida_as_20070226',
#     'caida_as_20070423',
#     'celegans_2019_hermaphrodite_chemical',
#     'add_health_comm27',
#     'add_health_comm40',
#     'add_health_comm61',
#     'add_health_comm73',
#     'add_health_comm81',
#     'advogato',
#     'bitcoin_alpha',
#     'caida_as_20040202',
#     'caida_as_20040503',
#     'caida_as_20040802',
#     'caida_as_20041101',
#     'caida_as_20050207',
#     'caida_as_20050502',
#     'caida_as_20050801',
#     'caida_as_20051107',
#     'caida_as_20060109',
#     'caida_as_20060130',
#     'caida_as_20060220',
#     'caida_as_20060313',
#     'caida_as_20060403',
#     'caida_as_20060424',
#     'caida_as_20060515',
#     'caida_as_20060605',
#     'caida_as_20060626',
#     'caida_as_20060717',
#     'caida_as_20060807',
#     'caida_as_20060828',
#     'caida_as_20060918',
#     'caida_as_20061009',
#     'caida_as_20061030',
#     'caida_as_20061120',
#     'caida_as_20061211',
#     'caida_as_20070101',
#     'caida_as_20070122',
#     'caida_as_20070212',
#     'caida_as_20070305',
#     'caida_as_20070917',
#     'celegans_2019_hermaphrodite_chemical_corrected',
#     'celegans_2019_hermaphrodite_chemical_synapse',
#     'celegans_2019_male_chemical_synapse',
#     'chicago_road',
#     'copenhagen_calls',
#     'dblp_cite',
#     'dutch_school_klas12b_net_3',
#     'dutch_school_klas12b_net_4m',
#     'ecoli_transcription_v1_1',
#     'fao_trade',
#     'fresh_webs_AkatoreA',
#     'fresh_webs_Blackrock',
#     'fresh_webs_Catlins',
#     'fresh_webs_DempstersAu',
#     'fresh_webs_German',
#     'fresh_webs_Lilkyeburn',
#     'fresh_webs_NorthCol',
#     'fresh_webs_SuttonAu',
#     'fresh_webs_Troy',
#     'freshman_t2',
#     'celegansneural',
#     'cintestinalis',
#     'copenhagen_sms',
#     'dutch_school_klas12b_net_1',
#     'dutch_school_klas12b_net_3m',
#     'dutch_school_klas12b_primary',
#     'email_company',
#     'foodweb_baywet',
#     'fresh_webs_AkatoreB',
#     'fresh_webs_Broad',
#     'fresh_webs_Coweeta1',
#     'fresh_webs_DempstersSp',
#     'fresh_webs_Healy',
#     'fresh_webs_Martins',
#     'fresh_webs_Powder',
#     'fresh_webs_SuttonSp',
#     'fresh_webs_Venlaw',
#     'freshman_t3',
#     'celegans_2019_male_chemical_corrected',
#     'chess',
#     'college_freshmen',
#     'cora',
#     'dutch_school_klas12b_net_2',
#     'dutch_school_klas12b_net_4',
#     'ecoli_transcription_v1_0',
#     'faa_routes',
#     'foodweb_little_rock',
#     'fresh_webs_Berwick',
#     'fresh_webs_Canton',
#     'fresh_webs_Coweeta17',
#     'fresh_webs_DempstersSu',
#     'fresh_webs_Kyeburn',
#     'fresh_webs_Narrowdale',
#     'fresh_webs_Stony',
#     'fresh_webs_SuttonSu',
#     'freshman_t0',
#     'freshman_t5',
#     'freshman_t6',
#     'freshmen_t3',
#     'genetic_multiplex_Arabidopsis',
#     'genetic_multiplex_Celegans',
#     'genetic_multiplex_Gallus',
#     'genetic_multiplex_HumanHerpes4',
#     'genetic_multiplex_Plasmodium',
#     'genetic_multiplex_Xenopus',
#     'gnutella_08',
#     'hens',
#     'inploid',
#     'jdk',
#     'macaques',
#     'moreno_taro',
#     'physician_trust',
#     'qa_user_mathoverflow_c2a',
#     'rhesus_monkey',
#     'un_migrations',
#     'us_agencies_alaska',
#     'us_agencies_california',
#     'us_agencies_delaware',
#     'us_agencies_hawaii',
#     'us_agencies_indiana',
#     'us_agencies_kentucky',
#     'us_agencies_maryland',
#     'us_agencies_minnesota',
#     'us_agencies_montana',
#     'us_agencies_newhampshire',
#     'us_agencies_newyork',
#     'us_agencies_ohio',
#     'us_agencies_pennsylvania',
#     'us_agencies_southdakota',
#     'us_agencies_utah',
#     'us_agencies_washington',
#     'us_agencies_wyoming',
#     'webkb_webkb_texas_link1',
#     'wiki_talk_br',
#     'wiki_talk_gl',
#     'wiki_talk_oc',
#     'word_adjacency_french',
#     'yeast_transcription',
#     'freshmen_t0',
#     'freshmen_t5',
#     'genetic_multiplex_Bos_Multiplex_Genetic',
#     'genetic_multiplex_DanioRerio',
#     'genetic_multiplex_HepatitusCVirus',
#     'genetic_multiplex_Mus',
#     'genetic_multiplex_Rattus',
#     'gnutella_04',
#     'gnutella_09',
#     'high_tech_company',
#     'interactome_figeys',
#     'jung',
#     'messal_shale',
#     'openflights',
#     'polblogs',
#     'qa_user_mathoverflow_c2q',
#     'sp_high_school_diaries',
#     'uni_email',
#     'us_agencies_arizona',
#     'us_agencies_colorado',
#     'us_agencies_florida',
#     'us_agencies_idaho',
#     'us_agencies_iowa',
#     'us_agencies_louisiana',
#     'us_agencies_massachusetts',
#     'us_agencies_mississippi',
#     'us_agencies_nebraska',
#     'us_agencies_newjersey',
#     'us_agencies_northcarolina',
#     'us_agencies_oklahoma',
#     'us_agencies_rhodeisland',
#     'us_agencies_tennessee',
#     'us_agencies_vermont',
#     'us_agencies_westvirginia',
#     'us_air_traffic',
#     'webkb_webkb_washington_link1',
#     'wiki_talk_cy',
#     'wiki_talk_ht',
#     'wikipedia_link_si',
#     'word_adjacency_japanese',
#     'freshmen_t2',
#     'freshmen_t6',
#     'genetic_multiplex_Candida',
#     'genetic_multiplex_Drosophila',
#     'genetic_multiplex_HumanHIV1',
#     'genetic_multiplex_Oryctolagus',
#     'genetic_multiplex_Sacchpomb',
#     'gnutella_06',
#     'gnutella_25',
#     'highschool',
#     'interactome_stelzl',
#     'law_firm',
#     'moreno_sheep',
#     'packet_delays',
#     'qa_user_mathoverflow_a2q',
#     'residence_hall',
#     'sp_high_school_survey',
#     'us_agencies_alabama',
#     'us_agencies_arkansas',
#     'us_agencies_connecticut',
#     'us_agencies_georgia',
#     'us_agencies_illinois',
#     'us_agencies_kansas',
#     'us_agencies_maine',
#     'us_agencies_michigan',
#     'us_agencies_missouri',
#     'us_agencies_nevada',
#     'us_agencies_newmexico',
#     'us_agencies_northdakota',
#     'us_agencies_oregon',
#     'us_agencies_southcarolina',
#     'us_agencies_texas',
#     'us_agencies_virginia',
#     'us_agencies_wisconsin',
#     'webkb_webkb_cornell_link1',
#     'webkb_webkb_wisconsin_link1',
#     'wiki_talk_eo',
#     'wiki_talk_nds',
#     'word_adjacency_darwin',
#     'word_adjacency_spanish'
# ]

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