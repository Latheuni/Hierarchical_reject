import os
import sys
import csv
import pandas as pd
import numpy as np
#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#import rpy2.robjects.packages as rpackages
#from rpy2.robjects.conversion import localconverter

def convertLabels_Flycell_head(LabelPath, FBBT_dfPath):
    FBBT_df = pd.read_csv(FBBT_dfPath)
    Labels = pd.read_csv(LabelPath)
    
    convert_dict = dict()
    # Get annotations up until anatomical structure and with correct seperator
    for i in range(0, FBBT_df.shape[0]-1):
        convert_dict[FBBT_df['X1'][i]] =  'root' + ';' + ';'.join(FBBT_df['X4'][i].split('_')[:FBBT_df['X4'][i].split('_').index('anatomical structure')][::-1])  + ';' +  FBBT_df['X1'][i]

    convert_dict['unannotated'] = "Needs to be filtered out"
    convert_dict['adult lamina epithelial/marginal glial cell'] = "Needs to be filtered out" # disagreement with hierarchy
    
    # Manual annotation so no hierarchy
    convert_dict['lamina intrinsic amacrine neuron Lai'] = "Needs to be filtered out" 
    convert_dict['artefact'] = "Needs to be filtered out" 
    convert_dict['Poxn neuron'] = "Needs to be filtered out" 
    convert_dict['photoreceptor-like'] = "Needs to be filtered out" 
    # Problem with multitude label
    convert_dict['T neuron T4/T5'] = 'root;cell;somatic cell;biological entity;anatomical entity;Thing;continuant;independent continuant;material entity;material anatomical entity;connected anatomical structure;cell;neuron;neuron;cholinergic neuron;adult cholinergic neuron;T neuron T4/T5;unspecified T neuron T4/5'
    convert_dict['T neuron T4/T5a-b'] = 'root;cell;somatic cell;biological entity;anatomical entity;Thing;continuant;independent continuant;material entity;material anatomical entity;connected anatomical structure;cell;neuron;neuron;cholinergic neuron;adult cholinergic neuron;T neuron T4/5;T neuron T4/T5a-b'
    convert_dict['T neuron T4/T5c-d'] = 'root;cell;somatic cell;biological entity;anatomical entity;Thing;continuant;independent continuant;material entity;material anatomical entity;connected anatomical structure;cell;neuron;neuron;cholinergic neuron;adult cholinergic neuron;T neuron T4/5;T neuron T4/T5c-d'
    
    # End node in middle hierarchy
    convert_dict['Kenyon cell'] = 'root;cell;somatic cell;biological entity;anatomical entity;Thing;continuant;independent continuant;material entity;material anatomical entity;connected anatomical structure;cell;neuron;neuron;interneuron;local neuron;intrinsic neuron;mushroom body intrinsic neuron;Kenyon cell;unspecified Kenyon cell '
    convert_dict['ensheathing glial cell'] = 'root;cell;somatic cell;glial cell;CNS glial cell;neuropil associated CNS glial cell;ensheathing glial cell;unspecified ensheathing glial cell '
    
    for key, value in convert_dict.items():
        if value.startswith('root;cell;somatic cell;biological entity;anatomical entity;Thing;'):
            convert_dict[key] = value.replace('root;cell;somatic cell;biological entity;anatomical entity;Thing;continuant;independent continuant;material entity;material anatomical entity;connected anatomical structure;cell;neuron;neuron;','root;cell;somatic cell;neuron;')
            
        if value.startswith('root;multicellular structure;portion of tissue;Thing;continuant;'):
            convert_dict[key] = value.replace('root;multicellular structure;portion of tissue;Thing;continuant;independent continuant;','root;multicellular structure;portion of tissue;')

    
    # Convert Labels to correct format
    Labels_with_hierarchy = []
    for i in Labels['annotations.annotation']:
        Labels_with_hierarchy.append(convert_dict[i])
    Labels.insert(1, 'Labels in hclf format', Labels_with_hierarchy)
    
    return(Labels)

def Preprocessing_Flyatlas_head(DataPath, LabelPath, FBBT_dfPath):
    """Preprocessing function for the Flyhead dataset. 
    The hierarchical information is formatted and cell populations with less than 10 members are discarded.

    Parameters
    ----------
    DataPath : str
        Local path to the Flyhead dataset (loom file)
    LabelPath : str
        Local path to the labels of the Flyhead dataset (csv file)
    FBBT_dfPath : str
        Local path to the FBbt ontology terms linked to the labels of the Flyhead dataset (csv file)
    
    Returns
    -------
    Data: pandas dataframe
    Labels: list
    """
    # Read in the count matrix
    SCopeLoomR = rpackages.importr('SCopeLoomR')
    loom = SCopeLoomR.open_loom(DataPath, mode="r+")
    dgem = SCopeLoomR.get_dgem(loom) # genes in rows and cells in columns
    with localconverter(ro.default_converter + pandas2ri.converter):
            pdf_from_r_df = ro.conversion.rpy2py(dgem)
    #print(type(pdf_from_r_df)) # says numpy ndarray
    p_df_from_r_df = pd.DataFrame(pdf_from_r_df)
    print(p_df_from_r_df.head()) # should hopefully have gene and cell in column and row names, just has indices in cell names
    Data_init = p_df_from_r_df.T # get cell x gene matrix
    
    # Read in the Labels and convert to correct format
    Labels_init = convertLabels_Flycell_head(LabelPath, FBBT_dfPath) # Labels in correct format in column 'Labels in hclf format' in df
    
    # filter cell populations with less than 10 cells
    l2 = Labels_init['Labels in hclf format'].value_counts()
    removed_classes = l2.index.values[l2<10] #numpy.ndarray
    Cells_To_Keep = [i for i in range(len(Labels_init['Labels in hclf format'])) if not Labels['Labels in hclf format'][i] in removed_classes] # list with indices
    labels = Labels_init.iloc[Cells_To_Keep]
    data = Data_init.iloc[Cells_To_Keep] # equal to .iloc[cells_to_keep,:]
    
    # filter 'NEEDS TO BE FILTERED OUT'
    index_keep = [i != "Needs to be filtered out" for i in labels['Labels in hclf format']]
    Labels = labels[index_keep]
    Data = data[index_keep] 
    return(Data, Labels)


def convertLabels_Flycell_body(LabelPath, FBBT_dfPath):
    FBBT_df = pd.read_csv(FBBT_dfPath)
    Labels = pd.read_csv(LabelPath)
    
    convert_dict = dict()
    # Get annotations up until anatomical structure and with correct seperator
    for i in range(0, FBBT_df.shape[0]-1):
        convert_dict[FBBT_df['X1'][i]] =  'root' + ';' + ';'.join(FBBT_df['X4'][i].split('_')[:FBBT_df['X4'][i].split('_').index('anatomical structure')][::-1])  + ';' +  FBBT_df['X1'][i]

    # make some specific changes in dict
    ## Sepecify teh to be filtered labels
    convert_dict['artefact'] = 'NEEDS TO BE FILTERED OUT'
    convert_dict['unannotated'] = 'NEEDS TO BE FILTERED OUT'
    convert_dict[FBBT_df['X1'][29]] = 'NEEDS TO BE FILTERED OUT'
    ## At intermediate nodes where there are some samples situated
    convert_dict['germline cell'] = 'root;cell;germline cell;general germline cell'
    convert_dict['follicle cell'] = 'root;cell;somatic cell;epithelial cell;follicle cell;general follicle cell'
    convert_dict['epithelial cell'] = 'root;cell;somatic cell;epithelial cell;general epithelial cell'
    convert_dict['muscle cell'] = 'root;cell;somatic cell;muscle cell:general muscle cell'

    ## Shorten some subdivisions when they get out of hand for clarity in classification
    convert_dict['spermatocyte'] = 'root;cell;germline cell;male-specific anatomical entity;male germline cell;spermatocyte'
    convert_dict['enteroendocrine cell'] = 'root;cell;endocrine cell;enteroendocrine cell'
    convert_dict['gustatory receptor neuron'] =  'root;cell;somatic cell;neuron;chemosensory neuron;gustatory receptor neuron'
    convert_dict['scolopidial neuron'] = 'root;cell;somatic cell;neuron;scolopidial neuron'
    convert_dict['leg taste bristle chemosensory neuron'] = 'root;cell;somatic cell;neuron;leg taste bristle chemosensory neuron'
    convert_dict['multidendritic neuron'] = 'root;cell;somatic cell;neuron;sensory neuron;multidendritic neuron'
    convert_dict['leg muscle motor neuron'] = 'root;cell;somatic cell;neuron;motor neuron;leg muscle motor neuron'
    convert_dict['adult fat body'] = 'root;multicellular structure;portion of tissue;fat body;adult fat body'
    convert_dict['perineurial glial sheath'] = 'root;multicellular structure;portion of tissue;connected anatomical structure;perineurial glial sheath'
    convert_dict['male accessory gland'] = 'root;multicellular structure;Thing;gland;male accessory gland'

    # These Labels have a different annotation in the annotation or orthology_ids
    convert_dict['follicle cell St. 9+'] = 'root;cell;somatic cell;epithelial cell;follicle cell;general follicle cell'
    
    # Convert Labels to correct format
    Labels_with_hierarchy = []
    for i in Labels['annotations.annotation']:
        Labels_with_hierarchy.append(convert_dict[i])
    Labels.insert(1, 'Labels in hclf format', Labels_with_hierarchy)
    
    return(Labels)

def Preprocessing_Flyatlas_body(DataPath, LabelPath, FBBT_dfPath):
    """Preprocessing function for the Flybody dataset. 
    
    The hierarchical information is formatted and cell populations with less than 10 members are discarded.

    Parameters
    ----------
    DataPath : str
        Local path to the Flybody dataset (loom file)
    LabelPath : str
        Local path to the labels of the Flybody dataset (csv file)
    FBBT_dfPath : str
        Local path to the FBbt ontology terms linked to the labels of the Flybody dataset (csv file)
    
    Returns
    -------
    Data: pandas dataframe
    Labels: list
    """
    
    # Read in the count matrix
    SCopeLoomR = rpackages.importr('SCopeLoomR')
    loom = SCopeLoomR.open_loom(DataPath, mode="r+")
    dgem = SCopeLoomR.get_dgem(loom) # genes in rows and cells in columns
    with localconverter(ro.default_converter + pandas2ri.converter):
            pdf_from_r_df = ro.conversion.rpy2py(dgem)
    #print(type(pdf_from_r_df)) # says numpy ndarray
    p_df_from_r_df = pd.DataFrame(pdf_from_r_df)
    print(p_df_from_r_df.head()) # should hopefully have gene and cell in column and row names, just has indices in cell names
    Data_init = p_df_from_r_df.T # get cell x gene matrix
    
    # Read in the Labels and convert to correct format
    Labels_init = convertLabels_Flycell_body(LabelPath, FBBT_dfPath) # Labels in correct format in column 'Labels in hclf format' in df
    
    # filter cell populations with less than 10 cells
    l2 = Labels_init['Labels in hclf format'].value_counts()
    removed_classes = l2.index.values[l2<10] #numpy.ndarray
    Cells_To_Keep = [i for i in range(len(Labels_init['Labels in hclf format'])) if not Labels['Labels in hclf format'][i] in removed_classes] # list with indices
    labels = Labels_init.iloc[Cells_To_Keep]
    data = Data_init.iloc[Cells_To_Keep] # equal to .iloc[cells_to_keep,:]
    
    # filter 'NEEDS TO BE FILTERED OUT'
    index_keep = [i != 'NEEDS TO BE FILTERED OUT' for i in labels['Labels in hclf format']]
    Labels = labels[index_keep]
    Data = data[index_keep]
    
    return(Data, Labels)