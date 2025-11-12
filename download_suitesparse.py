#!/usr/bin/env python3
"""
download_suitesparse.py

Usage:
  python3 download_suitesparse.py bcsstk31

This will:
 - open https://sparse.tamu.edu/MM/*/matrixname
 - parse the page to find the actual .mtx or .tar.gz download link
 - download and (if needed) extract the .mtx file to the current directory
"""

import sys
import requests
from bs4 import BeautifulSoup
import os
import shutil
import subprocess

if len(sys.argv) < 2:
    print("Usage: python3 download_suitesparse.py <matrix_name> [output_dir]")
    sys.exit(1)

matrix = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else "."
base = "https://sparse.tamu.edu/MM"

# Try to find the matrix page by scanning groups (quick approach: search site for matrix)
# We'll first try to use the direct page where group is unknown: site provides directory listing.
# So we'll search the site for the matrix name.
search_url = f"https://sparse.tamu.edu/cgi-bin/mmsearch?search={matrix}"
print("Searching SuiteSparse collection for", matrix)
r = requests.get(search_url, timeout=30)
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")

# look for links that point to /MM/.../<matrix>.*
download_page = None
for a in soup.find_all("a", href=True):
    href = a['href']
    if href.endswith(matrix) or href.endswith(matrix + "/"):
        download_page = href
        break
# fallback: find any link that mentions matrix name
if not download_page:
    for a in soup.find_all("a", href=True):
        if matrix in a['href']:
            download_page = a['href']
            break

if not download_page:
    print("Could not find matrix page via search. Trying direct group scan on main MM index may be needed.")
    # As fallback try brute-force groups list (less reliable). We'll try many groups.
    # To keep script simple, exit and instruct user to open the site in browser.
    print("Please open https://sparse.tamu.edu and search for the matrix, then paste the matrix page URL.")
    sys.exit(2)

# Make full URL if relative
if download_page.startswith("/"):
    matrix_page_url = "https://sparse.tamu.edu" + download_page
elif download_page.startswith("http"):
    matrix_page_url = download_page
else:
    matrix_page_url = "https://sparse.tamu.edu/" + download_page

print("Matrix page URL:", matrix_page_url)

print("Fetching matrix page...")
r = requests.get(matrix_page_url, timeout=30)
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")

# Look for links to .mtx .mtx.gz .tar.gz raw files
link = None
for a in soup.find_all("a", href=True):
    href = a['href']
    if href.endswith(".mtx") or href.endswith(".mtx.gz") or href.endswith(".tar.gz"):
        link = href
        break

if not link:
    # sometimes page has javascript or different layout. try to find any href containing ".mtx"
    for a in soup.find_all("a", href=True):
        if ".mtx" in a['href'] or ".tar.gz" in a['href']:
            link = a['href']
            break

if not link:
    print("Could not auto-locate a .mtx or .tar.gz link on the page. Please open the matrix page and copy the raw download URL.")
    sys.exit(3)

# Normalize link
if link.startswith("/"):
    link = "https://sparse.tamu.edu" + link
elif link.startswith("//"):
    link = "https:" + link
elif not link.startswith("http"):
    # relative path
    from urllib.parse import urljoin
    link = urljoin(matrix_page_url, link)

print("Found download link:", link)
fname = os.path.basename(link.split("?")[0])
outpath = os.path.join(outdir, fname)
print("Downloading to", outpath)

with requests.get(link, stream=True, timeout=60) as r:
    r.raise_for_status()
    with open(outpath, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

print("Download complete:", outpath)

# If tar.gz, extract; if .mtx.gz, gunzip; if .mtx, done
if outpath.endswith(".tar.gz") or outpath.endswith(".tgz"):
    print("Extracting tar.gz...")
    subprocess.check_call(["tar", "-xvzf", outpath, "-C", outdir])
    print("Extraction done.")
elif outpath.endswith(".mtx.gz"):
    print("Uncompressing .mtx.gz...")
    subprocess.check_call(["gunzip", "-f", outpath])
    print("Done. MTX file available.")
else:
    print("Downloaded file:", outpath)
