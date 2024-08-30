#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import requests
import urllib.parse
from tqdm import tqdm
from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry)
import configparser
import subprocess
import csv
from tqdm import tqdm
import ast


config_reader = configparser.ConfigParser()
config_reader.read('configurationWaterQSat.ini')

if config_reader.getboolean('RUN','download_section'):
    # Define your username and password
    username = config_reader['Download']['User']  # Replace with your actual username
    password = config_reader['Download']['pass']  # Replace with your actual password

    # Define global variables for token and expiry
    keycloak_token = None
    token_expiry = None

    # Define a function to get a Keycloak token
    def get_keycloak_token(username: str, password: str) -> str:
        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        }
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Response from the server was: {r.json()}"
            )
        response_json = r.json()
        global token_expiry
        # Token expires in seconds
        token_expiry = datetime.datetime.utcnow() + datetime.timedelta(seconds=response_json["expires_in"])
        return response_json["access_token"]

    # Define a function to get a valid Keycloak token
    def get_valid_keycloak_token() -> str:
        global keycloak_token, token_expiry
        if keycloak_token is None or datetime.datetime.utcnow() >= token_expiry:
            keycloak_token = get_keycloak_token(username, password)
        return keycloak_token

    def download_image(date_str):
        keycloak_token = get_valid_keycloak_token()
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        daydiff = int(config_reader['Download']['Delay'])
        start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%dT00:00:00.000Z")
        end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%dT00:00:00.000Z")

        aoi = config_reader['Download']['aoi']
        encoded_aoi = urllib.parse.quote(aoi)

        ###
        if config_reader['Download']['Mission'] == 'SENTINEL-2':
            request_url = (
                f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                f"$filter=Collection/Name eq 'SENTINEL-2' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{encoded_aoi}') and "
                f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date}"
            )

        elif config_reader['Download']['Mission'] == 'SENTINEL-3':

            request_url = (
                f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                f"$filter=Collection/Name eq 'SENTINEL-3' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{encoded_aoi}') and "
                f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date}"
            )

        elif config_reader['Download']['Mission'] == 'LANDSAT-8':

            request_url = (
                f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                f"$filter=Collection/Name eq 'LANDSAT-8' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{encoded_aoi}') and "
                f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date}"
            )
        
        # Make request and handle response
        response = requests.get(request_url)
        response.raise_for_status()
        json_response = response.json()

    
        valid_substrings = ast.literal_eval(config_reader['Missionconfigurations']['valid_substrings'])
        filtered_data = [
            item for item in json_response.get("value", [])
            if "Name" in item and all(substring in item["Name"] for substring in valid_substrings)
        ]

       
        if not filtered_data:
            print(f"No images found for date {date_str}")
            return

        # Calculate the time difference between the item OriginDate and the measurement date
        filtered_data = sorted(
            filtered_data, 
            key=lambda item: abs(datetime.datetime.strptime(item["OriginDate"], "%Y-%m-%dT%H:%M:%S.%fZ") - date)
        )

        # The first item in the sorted list has the minimum time difference
        closest_item = filtered_data[0]

        # Convert the closest item to a DataFrame for further processing
        filtered_df = pd.DataFrame([closest_item])
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {keycloak_token}"})

        for index, row in filtered_df.iterrows():
            base_download_path = config_reader['Download']['download_path']
            download_path = fr"{base_download_path}{row['Name']}.zip"

            # Check if file already exists
            if os.path.exists(download_path):
                print(f"File {row['Name']}.zip already exists. Skipping download.")
                continue

            url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({row['Id']})/$value"
            response = session.get(url, stream=True, allow_redirects=False)

            while response.status_code in (301, 302, 303, 307):
                url = response.headers["Location"]
                response = session.get(url, stream=True, allow_redirects=False)

            total_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024
            
        

            with open(download_path, "wb") as p:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {row['Name']}") as progress_bar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        p.write(data)
                        progress_bar.update(len(data))

    # Read the CSV file into a DataFrame
    csv_path = config_reader['Download']['CSV_date_path']
    csv_path = fr"{csv_path}"
    df = pd.read_csv(csv_path)


    # Create a new DataFrame with unique values in the "SDate" column
    new_df = pd.DataFrame({'SDate': df['Date']}).drop_duplicates()

    # Download images for each unique date in new_df
    for date in new_df['SDate']:
        try:
            download_image(str(date))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("Token expired. Refreshing token...")
                # Obtain a new token and retry the download
                get_valid_keycloak_token()
                download_image(str(date))
            else:
                print(f"Error downloading image for date {str(date)}: {str(e)}")
                
                
                
                
  
#Atomospheric correction with Sen2Cor

if config_reader.getboolean('RUN','AC_section'):
    sen2cor_path = config_reader['AC']['sen2cor_path']
    sen2cor_path = fr"{sen2cor_path}"
    # Path to Sen2Cor executable
    sen2cor_path = sen2cor_path

    csv_file_path = config_reader['AC']['csv_file_path']
    csv_file_path = fr"{csv_file_path}"

    # Open and read the CSV file
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        total_rows = sum(1 for _ in csv_reader)
        file.seek(0)


        for row in tqdm(csv_reader, total=total_rows, unit=" SAFE folder"):
            # Get the full path to the SAFE folder from the CSV
            full_path_to_safe_folder = row[0]
            
            if config_reader['AC']['operating_system'] == 'Linux':
                sen2cor_command = f"{sen2cor_path}/bin/L2A_Process {full_path_to_safe_folder}"
                
            elif config_reader['AC']['operating_system'] == 'Windows':
                sen2cor_command = f"cd /d {sen2cor_path} && L2A_Process.bat {full_path_to_safe_folder}"

            try:
                # Run the command
                subprocess.run(sen2cor_command, shell=True, check=True)
                # Wait for the process to complete
                subprocess.run("pause", shell=True)

            except subprocess.CalledProcessError as e:
                print(f"Error{full_path_to_safe_folder}: {e}")



