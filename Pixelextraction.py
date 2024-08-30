#!/usr/bin/env python
# coding: utf-8

# In[19]:

import configparser
import json
import ast

config_reader = configparser.ConfigParser()
config_reader.read('configurationWaterQSat.ini')



if config_reader.getboolean('RUN','Reflectance_Value_Extraction'):
    
    
    import numpy as np
    import pandas as pd
    import datetime
    import os
    import json
    import rasterio
    from rasterio.crs import CRS
    import pyproj
    from pyproj import transform,Proj
    from pyproj import CRS, Transformer
    from tqdm import tqdm
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from shapely import wkt
    import geopandas as gpd
    from shapely.geometry import Polygon
    import ee
    #ee.Authenticate()
    ee.Initialize()
    




    csv_path = config_reader['RFValue']['WQdatasetCSV']
    csv_path = fr"{csv_path}"
    finaldataset = pd.read_csv(csv_path)
    finaldataset["Date"] = pd.to_datetime(finaldataset["Date"],format= "%Y-%m-%d")
    finaldataset.insert(finaldataset.columns.get_loc('Date') + 1, 'Sat_Date', None)
    roi_str = config_reader['RFValue']['ROI']
    roi_dict = json.loads(roi_str)
    
    

    

    
    data_folder = config_reader['RFValue']['LVL2folder']
    data_folder = fr"{data_folder}"

    ###################################Pixel base extraction#############################################
    if config_reader['RFValue']['Scenario'] == 'PixelS2':
            
        band_names = ['SCL']

        for index, row in tqdm(finaldataset.iterrows(),total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))


            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date
                    

                if l2a_folder:
                    l2a_path = os.path.join(data_folder,l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path,"GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_20m.jp2"):
                                    jp2_file = os.path.join(root,file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                jp2_crs = src.crs

                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])

                                valuesC = list(src.sample([(lon, lat)]))


                                values = [valuesC[0][0]]
                                scl_filter_list = json.loads(config_reader['RFValue']['SCLfilter'])
                                if any(value in scl_filter_list for value in values):
                                    SCL_status = None
                                else:
                                    SCL_status = 1

                                # Update the DataFrame with the pixel value
                                finaldataset.at[index, band_name] = SCL_status
                        else:
                            print(f"JP2 file not found for {date} and band {band_name}")
                else:
                    print(f"L2A folder not found for {date}")
            else:
                print(f"Invalid ROI number: {roi_number}")


        finaldataset.dropna(subset="SCL",inplace=True)
        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("SCL filter Done.")
        


        band_names = json.loads(config_reader['RFValue']['S2Bands20m'])
        
        for index, row in tqdm(finaldataset.iterrows(),total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                            
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                if l2a_folder:
                    l2a_path = os.path.join(data_folder,l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path,"GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_20m.jp2"):
                                    jp2_file = os.path.join(root,file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                jp2_crs = src.crs
                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])
                                valuesC = list(src.sample([(lon, lat)]))
                                average_value = (valuesC[0][0] * 0.0001)
                                finaldataset.at[index, band_name] = round(average_value, 6)
                        else:
                            continue
                else:
                    continue
            else:
                continue

        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("20m resolution bands done.")

        band_names =  json.loads(config_reader['RFValue']['S2Bands60m'])

        # Define the target resolution
        target_resolution = int(config_reader['RFValue']['S2Bands60mresampling'])

        for index, row in tqdm(finaldataset.iterrows(), total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                if l2a_folder:
                    l2a_path = os.path.join(data_folder, l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path, "GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_60m.jp2"):
                                    jp2_file = os.path.join(root, file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                # Resample data to target resolution
                                scale_factor = src.res[0] / target_resolution
                                new_height = int(src.height * scale_factor)
                                new_width = int(src.width * scale_factor)

                                data = src.read(
                                    out_shape=(src.count, new_height, new_width),
                                    resampling=Resampling.bilinear
                                )

                                # Scale image transform to the new dimensions
                                transform = src.transform * src.transform.scale(
                                    (src.width / data.shape[-1]),
                                    (src.height / data.shape[-2])
                                )

                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])
                                valuesC = list(src.sample([(lon, lat)]))
                                average_value = (valuesC[0][0] * 0.0001)

                                # Update the DataFrame with the pixel value
                                finaldataset.at[index, band_name] = round(average_value, 6)
                        else:
                            continue
                else:
                    continue         
            else:
                continue

        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("60m resolution bands done.")
        print("Done!")
        
        
    ###################################three by three extraction#############################################        
        
    if config_reader['RFValue']['Scenario'] == 'threebythreeS2':
            
        band_names = ['SCL']

        for index, row in tqdm(finaldataset.iterrows(),total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                if l2a_folder:
                    l2a_path = os.path.join(data_folder,l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path,"GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_20m.jp2"):
                                    jp2_file = os.path.join(root,file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                jp2_crs = src.crs

                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])


                                valuesUL = list(src.sample([(lon-20, lat+20)]))
                                valuesU = list(src.sample([(lon, lat+20)]))
                                valuesUR = list(src.sample([(lon+20, lat+20)]))
                                valuesL = list(src.sample([(lon-20, lat)]))
                                valuesC = list(src.sample([(lon, lat)]))
                                valuesR = list(src.sample([(lon+20, lat)]))
                                valuesDL = list(src.sample([(lon-20, lat-20)]))
                                valuesD = list(src.sample([(lon, lat-20)]))
                                valuesDR = list(src.sample([(lon+20, lat-20)]))

                                threebythreepixelconsideration = ast.literal_eval(config_reader['RFValue']['threebythreepixelconsideration'])

                                                                
                                pixel_values = {
                                    "valuesUL[0][0]": valuesUL[0][0],
                                    "valuesU[0][0]": valuesU[0][0],
                                    "valuesUR[0][0]": valuesUR[0][0],
                                    "valuesL[0][0]": valuesL[0][0],
                                    "valuesC[0][0]": valuesC[0][0],
                                    "valuesR[0][0]": valuesR[0][0],
                                    "valuesDL[0][0]": valuesDL[0][0],
                                    "valuesD[0][0]": valuesD[0][0],
                                    "valuesDR[0][0]": valuesDR[0][0],
                                }
                                
                                
                                values = [pixel_values.get(pixel, None) for pixel in threebythreepixelconsideration]

                                scl_filter_list = json.loads(config_reader['RFValue']['SCLfilter'])
                                if any(value in scl_filter_list for value in values):
                                    SCL_status = None
                                else:
                                    SCL_status = 1

                                # Update the DataFrame with the pixel value
                                finaldataset.at[index, band_name] = SCL_status
                        else:
                            continue
                else:
                    continue
            else:
                continue


        finaldataset.dropna(subset="SCL",inplace=True)
        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("SCL filter done.")
        
        
        

        band_names = json.loads(config_reader['RFValue']['S2Bands20m'])
        

        for index, row in tqdm(finaldataset.iterrows(),total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                if l2a_folder:
                    l2a_path = os.path.join(data_folder,l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path,"GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_20m.jp2"):
                                    jp2_file = os.path.join(root,file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                jp2_crs = src.crs


                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])


                                valuesUL = list(src.sample([(lon-20, lat+20)]))
                                valuesU = list(src.sample([(lon, lat+20)]))
                                valuesUR = list(src.sample([(lon+20, lat+20)]))
                                valuesL = list(src.sample([(lon-20, lat)]))
                                valuesC = list(src.sample([(lon, lat)]))
                                valuesR = list(src.sample([(lon+20, lat)]))
                                valuesDL = list(src.sample([(lon-20, lat-20)]))
                                valuesD = list(src.sample([(lon, lat-20)]))
                                valuesDR = list(src.sample([(lon+20, lat-20)]))

                                threebythreepixelconsideration = ast.literal_eval(config_reader['RFValue']['threebythreepixelconsideration'])

                                                                
                                pixel_values = {
                                    "valuesUL[0][0]": valuesUL[0][0],
                                    "valuesU[0][0]": valuesU[0][0],
                                    "valuesUR[0][0]": valuesUR[0][0],
                                    "valuesL[0][0]": valuesL[0][0],
                                    "valuesC[0][0]": valuesC[0][0],
                                    "valuesR[0][0]": valuesR[0][0],
                                    "valuesDL[0][0]": valuesDL[0][0],
                                    "valuesD[0][0]": valuesD[0][0],
                                    "valuesDR[0][0]": valuesDR[0][0],
                                }
                                
                                
                                values = [pixel_values.get(pixel, None) for pixel in threebythreepixelconsideration]


                                average_value = np.nanmean(values) * 0.0001

                                # Update the DataFrame with the pixel value
                                finaldataset.at[index, band_name] = round(average_value, 6)
                        else:
                            continue
                else:
                    continue
            else:
                continue


        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("20m resolution bands done.")

        band_names =  json.loads(config_reader['RFValue']['S2Bands60m'])

        # Define the target resolution
        target_resolution = int(config_reader['RFValue']['S2Bands60mresampling'])

        for index, row in tqdm(finaldataset.iterrows(), total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                        
                if l2a_folder:
                    l2a_path = os.path.join(data_folder, l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path, "GRANULE")

                    for band_name in band_names:
                        jp2_file = None
                        for root, _, files in os.walk(jp2_files_folder):
                            for file in files:
                                if file.endswith(f"_{band_name}_60m.jp2"):
                                    jp2_file = os.path.join(root, file)
                                    break
                            if jp2_file:
                                break

                        if jp2_file:
                            with rasterio.open(jp2_file) as src:
                                # Resample data to target resolution
                                scale_factor = src.res[0] / target_resolution
                                new_height = int(src.height * scale_factor)
                                new_width = int(src.width * scale_factor)

                                data = src.read(
                                    out_shape=(src.count, new_height, new_width),
                                    resampling=Resampling.bilinear
                                )

                                # Scale image transform to the new dimensions
                                transform = src.transform * src.transform.scale(
                                    (src.width / data.shape[-1]),
                                    (src.height / data.shape[-2])
                                )

                                wgs84 = CRS.from_epsg(4326)
                                transformer = Transformer.from_crs(wgs84, jp2_crs, always_xy=True)
                                lon, lat = transformer.transform(roi_geometry[0], roi_geometry[1])


                                valuesUL = list(src.sample([(lon-20, lat+20)]))
                                valuesU = list(src.sample([(lon, lat+20)]))
                                valuesUR = list(src.sample([(lon+20, lat+20)]))
                                valuesL = list(src.sample([(lon-20, lat)]))
                                valuesC = list(src.sample([(lon, lat)]))
                                valuesR = list(src.sample([(lon+20, lat)]))
                                valuesDL = list(src.sample([(lon-20, lat-20)]))
                                valuesD = list(src.sample([(lon, lat-20)]))
                                valuesDR = list(src.sample([(lon+20, lat-20)]))

                                threebythreepixelconsideration = ast.literal_eval(config_reader['RFValue']['threebythreepixelconsideration'])

                                                                
                                pixel_values = {
                                    "valuesUL[0][0]": valuesUL[0][0],
                                    "valuesU[0][0]": valuesU[0][0],
                                    "valuesUR[0][0]": valuesUR[0][0],
                                    "valuesL[0][0]": valuesL[0][0],
                                    "valuesC[0][0]": valuesC[0][0],
                                    "valuesR[0][0]": valuesR[0][0],
                                    "valuesDL[0][0]": valuesDL[0][0],
                                    "valuesD[0][0]": valuesD[0][0],
                                    "valuesDR[0][0]": valuesDR[0][0],
                                }
                                
                                
                                values = [pixel_values.get(pixel, None) for pixel in threebythreepixelconsideration]


                                average_value = np.nanmean(values) * 0.0001

                                # Update the DataFrame with the pixel value
                                finaldataset.at[index, band_name] = round(average_value, 6)
                        else:
                            continue
                else:
                    continue
            else:
                continue

        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("60m resolution bands done.")
        print("Done!")



    ###################################all area extraction#############################################

    if config_reader['RFValue']['Scenario'] == 'allareaS2':
            
        
        band_names = json.loads(config_reader['RFValue']['S2Bands20m'])

        # Read the WKT file
        WKTpath = config_reader['RFValue']['WKTpath']
        wkt_file = fr"{WKTpath}"
        with open(wkt_file, "r") as file:
            wkt_string = file.read()

        # Convert WKT to a Shapely geometry
        lake_geometry = wkt.loads(wkt_string)

        for index, row in tqdm(finaldataset.iterrows(), total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date

                        
                if l2a_folder:
                    l2a_path = os.path.join(data_folder, l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path, "GRANULE")

                    # Load the SCL band
                    scl_band_name = "SCL"
                    scl_jp2_file = None

                    for root, _, files in os.walk(jp2_files_folder):
                        for file in files:
                            if file.endswith(f"_{scl_band_name}_20m.jp2"):
                                scl_jp2_file = os.path.join(root, file)
                                break
                        if scl_jp2_file:
                            break

                    if scl_jp2_file:
                        with rasterio.open(scl_jp2_file) as scl_src:
                            scl_data = scl_src.read(1)
                            scl_transform = scl_src.transform
                            scl_crs = scl_src.crs

                            # Transform lake geometry to SCL CRS
                            wgs84 = CRS.from_epsg(4326)
                            transformer = Transformer.from_crs(wgs84, scl_crs, always_xy=True)
                            lake_geometry_transformed = transformer.transform(*lake_geometry.exterior.coords.xy)
                            lake_geometry_transformed = Polygon(zip(*lake_geometry_transformed))

                         
                            
                            scl_filter_list = json.loads(config_reader['RFValue']['SCLfilter'])
                            

                            # Create mask for the lake area
                            lake_mask = geometry_mask([lake_geometry_transformed], transform=scl_transform, invert=True, out_shape=scl_data.shape)

               
                            # Create mask where SCL value is not in the list of exclude_values
                            not_in_list_mask = ~np.isin(scl_data, scl_filter_list)

                            # Apply the lake mask to the not-in-list mask
                            water_mask = not_in_list_mask & lake_mask
                            
                            for band_name in band_names:
                                jp2_file = None
                                for root, _, files in os.walk(jp2_files_folder):
                                    for file in files:
                                        if file.endswith(f"_{band_name}_20m.jp2"):
                                            jp2_file = os.path.join(root, file)
                                            break
                                    if jp2_file:
                                        break

                                if jp2_file:
                                    with rasterio.open(jp2_file) as src:
                                        band_data = src.read(1)
                                        band_transform = src.transform
                                        band_crs = src.crs

                                        # Ensure the band data is in the same CRS as the SCL data
                                        if band_crs != scl_crs:
                                            print(f"CRS mismatch: {band_crs} vs {scl_crs}")
                                            continue

                                        # Apply the water mask to the band data
                                        water_band_data = band_data[water_mask]

                                        # Calculate the average value
                                        average_value = (water_band_data * 0.0001).mean()

                                        # Update the DataFrame with the pixel value
                                        finaldataset.at[index, band_name] = round(average_value, 6)
                                else:
                                    print(f"JP2 file not found for {date} and band {band_name}")
                    else:
                        print(f"SCL file not found for {date}")
                else:
                    print(f"L2A folder not found for {date}")      
            else:
                print(f"Invalid ROI number: {roi_number}")

        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("SCL filter done.")
                
        band_names =  json.loads(config_reader['RFValue']['S2Bands60m'])

        for index, row in tqdm(finaldataset.iterrows(), total=finaldataset.shape[0]):
            date = row["Date"]
            roi_number = row["ROI"]
            daydiff = int(config_reader['RFValue']['Delay'])
            start_date = (date - datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")
            end_date = (date + datetime.timedelta(days=daydiff)).strftime("%Y-%m-%d")

            roi_geometry = roi_dict.get(str(roi_number))

            if roi_geometry:
                folder_date = pd.to_datetime(date, format="%Y-%m-%d").strftime("%Y%m%d")
                l2a_folder = None
                closest_date = None
                min_date_diff = None

                for root, dirs, files in os.walk(data_folder):
                    for dir_name in dirs:
                        parts = dir_name.split("_")
                        if "MSIL2A" in parts:
                            date_parts = parts[2].split("T")[0]
                            folder_date_obj = pd.to_datetime(date_parts, format="%Y%m%d")

                            # Check if date_parts is within the start_date and end_date range
                            if start_date <= folder_date_obj.strftime("%Y-%m-%d") <= end_date:
                                date_diff = abs((folder_date_obj - date).days)

                                # Select the date closest to the original date
                                if min_date_diff is None or date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    closest_date = date_parts
                                    l2a_folder = dir_name

                    if l2a_folder:
                        break
                        
                if closest_date:
                    finaldataset.at[index, 'Sat_Date'] = closest_date
                    
                        

                if l2a_folder:
                    l2a_path = os.path.join(data_folder, l2a_folder)
                    jp2_files_folder = os.path.join(l2a_path, "GRANULE")

                    # Load the SCL band
                    scl_band_name = "SCL"
                    scl_jp2_file = None

                    for root, _, files in os.walk(jp2_files_folder):
                        for file in files:
                            if file.endswith(f"_{scl_band_name}_60m.jp2"):
                                scl_jp2_file = os.path.join(root, file)
                                break
                        if scl_jp2_file:
                            break

                    if scl_jp2_file:
                        with rasterio.open(scl_jp2_file) as scl_src:
                            scl_data = scl_src.read(1)
                            scl_transform = scl_src.transform
                            scl_crs = scl_src.crs

                            # Transform lake geometry to SCL CRS
                            wgs84 = CRS.from_epsg(4326)
                            transformer = Transformer.from_crs(wgs84, scl_crs, always_xy=True)
                            lake_geometry_transformed = transformer.transform(*lake_geometry.exterior.coords.xy)
                            lake_geometry_transformed = Polygon(zip(*lake_geometry_transformed))

   
                            
                            scl_filter_list = json.loads(config_reader['RFValue']['SCLfilter'])
                            

                            # Create mask for the lake area
                            lake_mask = geometry_mask([lake_geometry_transformed], transform=scl_transform, invert=True, out_shape=scl_data.shape)


                            not_in_list_mask = ~np.isin(scl_data, scl_filter_list)

                            water_mask = not_in_list_mask & lake_mask
                            
                            
                            


                            for band_name in band_names:
                                jp2_file = None
                                for root, _, files in os.walk(jp2_files_folder):
                                    for file in files:
                                        if file.endswith(f"_{band_name}_60m.jp2"):
                                            jp2_file = os.path.join(root, file)
                                            break
                                    if jp2_file:
                                        break

                                if jp2_file:
                                    with rasterio.open(jp2_file) as src:
                                        band_data = src.read(1)
                                        band_transform = src.transform
                                        band_crs = src.crs

                                        # Ensure the band data is in the same CRS as the SCL data
                                        if band_crs != scl_crs:
                                            print(f"CRS mismatch: {band_crs} vs {scl_crs}")
                                            continue

                                        # Apply the water mask to the band data
                                        water_band_data = band_data[water_mask]

                                        # Calculate the average value
                                        average_value = (water_band_data * 0.0001).mean()

                                        # Update the DataFrame with the pixel value
                                        finaldataset.at[index, band_name] = round(average_value, 6)
                                else:
                                    continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
        finaldataset.dropna(subset="B02",inplace=True)
        finaldataset['Sat_Date'] = pd.to_datetime(finaldataset['Sat_Date'], format="%Y-%m-%d")
        print("60m resolution bands done.")
        print("Done!")
        
        
    Savecsvpath = config_reader['RFValue']['Savecsvpath']
    Savecsvpath = fr"{Savecsvpath}"
    finaldataset.to_csv(Savecsvpath, index=False)
    print(f"File saved in {Savecsvpath}.")



# In[20]:


finaldataset


# In[ ]:




