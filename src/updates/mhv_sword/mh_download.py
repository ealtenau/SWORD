import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
import pandas as pd
import time

# Editable: HTTP Basic Auth credentials
USERNAME = "hydrography"
PASSWORD = "rivernetwork"

DOWNLOADABLE_EXTENSIONS = {".tar"}

###########################################################################

def is_download_link(href):
    if not href:
        return False
    return any(href.lower().endswith(ext) for ext in DOWNLOADABLE_EXTENSIONS)

###########################################################################

def get_download_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if is_download_link(href):
                full_url = urljoin(url, href)
                links.append(full_url)

        return links

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

###########################################################################

#no authorization required.
def download_file(url, folder="downloads"):
    os.makedirs(folder, exist_ok=True)
    local_filename = os.path.basename(urlparse(url).path)

    # If filename is blank, use a fallback
    if not local_filename:
        local_filename = "downloaded_file"

    local_path = os.path.join(folder, local_filename)

    try:
        print(f"Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to: {local_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

###########################################################################

#username and password required on download. 
def download_file_auth(url, folder="downloads"):
    os.makedirs(folder, exist_ok=True)
    local_filename = os.path.basename(urlparse(url).path) or "downloaded_file"
    local_path = os.path.join(folder, local_filename)

    try:
        print(f"Downloading: {url}")
        with requests.get(url, auth=(USERNAME, PASSWORD), stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to: {local_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

###########################################################################
###########################################################################
###########################################################################

if __name__ == "__main__":
    
    #file paths. 
    csv_fn = main_dir+'/data/inputs/MERIT_Hydro/AS_mh_files.csv'
    outdir = main_dir+'/data/inputs/MERIT_Hydro/'+os.path.basename(csv_fn)[-15:-13]+'/'
    if os.path.isdir(outdir) is False:
        os.makedirs(outdir)
        os.makedirs(outdir+'elv/')
        os.makedirs(outdir+'upa/')
        os.makedirs(outdir+'wth/')

    #files to download.
    dls_csv = pd.read_csv(csv_fn)
    tiles = np.array(dls_csv['zone'])
    
    #finding all download links on webpage. 
    url = "https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"
    downloads = get_download_links(url)

    #narrowing down the download links to elv, upa, and wth. 
    elv_dls = np.sort(np.array([file for file in downloads if 'elv_' in file]))
    upa_dls = np.sort(np.array([file for file in downloads if 'upa_' in file]))
    wth_dls = np.sort(np.array([file for file in downloads if 'wth_' in file]))
    
    #starting downloads.
    start = time.time()
    for t in list(range(len(tiles))):
        idx = [i for i, s in enumerate(elv_dls) if tiles[t] in s]

        print('Starting tile download:', tiles[t])
        try:
            download_file_auth(elv_dls[idx][0], outdir+'elv/')
        except:
            print('!!! elv download failed:', tiles[t])
        try:
            download_file_auth(upa_dls[idx][0], outdir+'upa/')
        except:
            print('!!! upa download failed:', tiles[t])
        try:
            download_file_auth(wth_dls[idx][0], outdir+'wth/')
        except:
            print('!!! wth download failed:', tiles[t])
    
    end = time.time()
    print(str(np.round((end-start)/60, 2)) + ' min')

