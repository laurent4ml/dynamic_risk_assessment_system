import logging
import subprocess
import re 

def outdated_packages_list():
    ''' get list of outdated packages '''
    outdated_packages = subprocess.check_output(['pip', 'list','--outdated'])
    
    with open('outdated.txt', 'wb') as f:
        f.write(outdated_packages)

    output = outdated_packages.decode()
    packages = list()
    for outdated_package in output.split("\n"):
        outdated_package = re.sub(' +', ' ', outdated_package)
        tabs = outdated_package.split()
        if len(tabs) > 2:
            pack = {'package': tabs[0], 'current': tabs[1], 'latest': tabs[2]}
            logging.info(f"row: {pack}")
            packages.append(pack)
    logging.info(packages)
    return packages 