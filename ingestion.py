import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def list_all_files(path):
    ''' 
    return files in a directory
    
    Args
        path: (str) directory storing input files

    Output
        file_list: (list) list of file contained in directory
    '''
    file_list = list()

    try:
        assert len(path) > 0, "len of path should not be null"
        
        assert os.path.exists(path), "path does not exists"

        for (_, _, files) in os.walk(path):
            for f in files:
                if '.csv' in f:
                    logger.info(f"list_all_files - file_name: {f}")
                    file_list.append(f)
    except AssertionError as msg: 
        logger.info(msg)

    return file_list

def condition(file_name, ingested_file_list):
    ''' 
    condition used in filter to process only files never ingested
    
    Args
        file_name: (str) file to search for
        ingested_file_list: (list) list of files to look into

    Output
        return True if the file_name is in file_list, False otherwise
    '''
    logger.info(f"condition - start")
    logger.info(f"condition - file_name: {file_name}")
    logger.info(f"condition - ingested_file_list: {str(ingested_file_list)}")
    if file_name in ingested_file_list:
        return False
    else:
        return True

def filter_new_files(ingested_file_name, all_files_in_directory):
    ''' 
    filter process to get only files not already processed

    Args
        ingested_file_name: (str) file containing all previously processed files
        all_files_in_directory: (list) all files in directory

    Output
        filtered_list: (list) list of files that are not part of the
        list of previously processed files
    '''
    logger.info(f"filter_new_files - ingested_file_name: {ingested_file_name}")

    filtered_list = list()

    try:
        assert isinstance(all_files_in_directory, list), "all_files_in_directory not a list"
    except AssertionError as m:
        logger.info(m)
        return filtered_list
    
    if not os.path.exists(ingested_file_name):
        with open(ingested_file_name, 'w') as f:
            logger.info(f"filter_new_files - new file created: {ingested_file_name}")

    with open(ingested_file_name, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
        logger.info(f"filter_new_files - lines: {str(lines)}")
        if not len(lines):
            filtered_list = all_files_in_directory
        else:
            for file in all_files_in_directory:
                if condition(file, lines):
                    filtered_list.append(file)

    logger.info(f"filter_new_files - filtered files: {str(filtered_list)}")
    return filtered_list

def store_new_files(file_name, new_files):
    ''' store/append a new file found in input directory
    in file used to keep list of files previously processed files

    Args
        file_name: (str) file to append new files to
        new_files: (list) files to be appended
    '''
    logger.info(f"store_new_files - file_name: {file_name}")

    try:
        assert len(file_name) > 0, "len of path should not be null"

        assert os.path.exists(file_name), "path does not exists"

        logger.info(f"store_new_files - new_files: {new_files}")
        for new_file in new_files:
            logger.info(f"store_new_files - new file: {new_file}")

        hs = open(file_name, "a")
        for file in new_files:
            logger.info(f"store_new_files - writing file: {file}")
            hs.write(file + "\n")
        hs.close()
    except AssertionError as msg:
        logger.info(msg)

# Function for data ingestion
def merge_multiple_dataframe(input_files, output_file, input_file_directory):
    '''
    check for datasets, compile them together, remove duplicates
    and write to an output file

    Args
        input_files: (list) list of files to compile
        output_file: (str) file to store compile data
        input_file_directory: (str) input file directory
    Output
        response: (bool) True if merge succeded False otherwise
    '''
    try:
        assert isinstance(input_files, list), "input_files not a list"

        assert isinstance(output_file, str), "output_file not defined"

        assert len(output_file), "output_file name missing"

        df = pd.DataFrame()

        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                logger.info(f"merge_multiple_dataframe - new file created: {output_file}")
        else:
            try:
                df = pd.read_csv(output_file)
            except pd.errors.EmptyDataError:
                logger.info(f"{output_file} is empty and has been skipped.")

        length = len(df)
        logger.info(f"merge_multiple_dataframe - length df: {length}")
        for input_file in input_files:
            input_path = os.path.join(os.getcwd(), input_file_directory, input_file)
            data = pd.read_csv(input_path)
            logger.info(f"merge_multiple_dataframe - adding {len(data)} rows")
            if length:
                try:
                    df.append(data)
                except ValueError as e:
                    logger.info(e)
                    raise ValueError(f"Error merging data: {e}")
            else:
                df = data
            
        logger.info(f"merge_multiple_dataframe - dropping duplicates")
        df = df.drop_duplicates()
        logger.info(f"merge_multiple_dataframe - store to output file")
        df.to_csv(output_file)
    except AssertionError as m:
        logging.info(m)
        return False
    return True


if __name__ == '__main__':
    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    logger.info("ingestion - list_all_files - start")
    all_files = list_all_files(os.getcwd() + "/" + input_folder_path)
    logger.info("list_all_files - end\n")

    logger.info("filter_new_files - start")
    ingested_file_name = os.getcwd() + "/" + output_folder_path + "/ingestedfiles.txt"
    new_files = filter_new_files(ingested_file_name, all_files)
    logger.info("filter_new_files - end\n")
    
    if len(new_files):
        logger.info("merge_multiple_dataframe - start")
        merge_response = merge_multiple_dataframe(
            new_files,
            os.getcwd() + "/" + output_folder_path + "/finaldata.csv",
            input_folder_path
        )
        logger.info("merge_multiple_dataframe - end")
        
        if merge_response:
            logger.info("store_new_file - start")
            store_new_files(ingested_file_name, new_files)
            logger.info("store_new_file - end")
        else:
            logger.info("error merging data")
