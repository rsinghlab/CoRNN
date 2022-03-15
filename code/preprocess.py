import torch
from torch.utils.data import Dataset
import numpy as np

def clean_data(data,corre):
    cleaned_data = []
    #calculate the correlation
    for idx in range(0,len(data),100):
        data_item = np.array(data[idx:idx+100])
        if corre != None:
            if corre > 0:
                data_cla_label = 1 if float(data_item[0][1])>0 else 0
                data_reg_label = float(data_item[0][1])
            else:
                data_cla_label = 0 if float(data_item[0][1])>0 else 1
                data_reg_label = -float(data_item[0][1])
        else:
            data_cla_label = 1 if float(data_item[0][1])>0 else 0
            data_reg_label = float(data_item[0][1])
        # print("corre: ",corre)
        # print("original: {}".format(data_item[0][1]))
        # print("now: {}, {}".format(data_cla_label,data_reg_label))
        # print("data item:",data_item)
        # print()
        data_item = data_item[:,3:]
        data_item = data_item.astype(np.float)
        cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"reg_label":data_reg_label})
    return cleaned_data

def clean_data_with_mean(data,corre,mean_evec,test_by_region = False):
    cleaned_data = []
    #calculate the correlation

    for idx in range(0,len(data),100):
        mean_id = int(idx/100)
        mean_value = mean_evec[mean_id]
        if test_by_region:
            mean_value = float(mean_value)
        else:
            if mean_value == "nan":
                print("find nan")
                print(mean_value)
                continue
            else:
                mean_value = float(mean_value)
        data_item = np.array(data[idx:idx+100])
        if corre != None:
            if corre > 0:
                data_cla_label = 1 if float(data_item[0][1])>0 else 0
                data_reg_label = float(data_item[0][1])
            else:
                data_cla_label = 0 if float(data_item[0][1])>0 else 1
                data_reg_label = -float(data_item[0][1])
        else:
            data_cla_label = 1 if float(data_item[0][1])>0 else 0
            data_reg_label = float(data_item[0][1])
        data_item = data_item[:,3:]
        data_item = data_item.astype(np.float)

        cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"reg_label":data_reg_label,"mean_evec":mean_value})
    return cleaned_data
def clean_data_with_mean_and_evec(data,corre,mean_evec,test_by_region = False):
    cleaned_data = []
    #calculate the correlation

    for idx in range(0,len(data),100):
        mean_id = int(idx/100)
        mean_value = mean_evec[mean_id]
        if test_by_region:
            mean_value = float(mean_value)
        else:
            if mean_value == "nan":
                print("find nan")
                print(mean_value)
                continue
            else:
                mean_value = float(mean_value)
        data_item = np.array(data[idx:idx+100])
        if corre != None:
            if corre > 0:
                data_cla_label = 1 if float(data_item[0][1])>0 else 0
                data_reg_label = float(data_item[0][1])
            else:
                data_cla_label = 0 if float(data_item[0][1])>0 else 1
                data_reg_label = -float(data_item[0][1])
        else:
            data_cla_label = 1 if float(data_item[0][1])>0 else 0
            data_reg_label = float(data_item[0][1])
        data_item = data_item[:,3:]
        data_item = data_item.astype(np.float)

        cleaned_data.append({"data":data_item,"evec":float(data_item[0][1]),"cla_label":data_cla_label,"reg_label":data_reg_label,"mean_evec":mean_value})
    return cleaned_data

def combine_data(all_data):
    combined_data = []
    for key, item in all_data.items():
        # print(key)
        combined_data += item
    return combined_data

def combine_data_rf(all_data):
    combined_x = []
    combined_y = []
    print("600 features")
    for key, item in all_data.items():
        for data in item:
            # print("here")
            # print(data["data"].shape)
            flettened_x = np.array(data["data"]).flatten()
            # print(flettened_x.shape)
            # print(data["cla_label"])
            combined_x.append(flettened_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_rf_601(all_data):
    combined_x = []
    combined_y = []
    print("600 features")
    for key, item in all_data.items():
        for data in item:
            # print("here")
            # print(data["data"].shape)
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flettened_x = np.array(data["data"]).flatten()
            # print(flettened_x.shape)
            # print(data["cla_label"])
            appended_x = np.append(flatten_x,data["mean_evec"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y


def combine_data_avg(all_data):
    # 6 features (6 mean)
    print("6 features")
    combined_x = []
    combined_y = []

    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_x = mean_data.flatten()
            # print(flatten_x.shape)
            combined_x.append(flatten_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_with_mean_avg(all_data):
    #7 features (6 mean + mean evec)
    print("6 features + mean")
    combined_x = []
    combined_y = []

    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_x = mean_data.flatten()
            appended_x = np.append(flatten_x,data["mean_evec"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_avg_std(all_data):
    # 12 features (6 mean + 6 std)
    print("12 features")
    combined_x = []
    combined_y = []

    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_mean = mean_data.flatten()

            std_data = np.std(np.array(data["data"]), axis=0)
            flatten_std = std_data.flatten()

            concat_mean_std = np.concatenate((flatten_mean,flatten_std))

            combined_x.append(concat_mean_std)
            combined_y.append(data["cla_label"])
    # print(combined_x[0].shape)
    return combined_x, combined_y

def combine_data_with_mean_avg_std(all_data):
    # 13 feature, (6 mean + 6 std +1 mean evec)
    print("12 features + mean")
    combined_x = []
    combined_y = []

    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_mean = mean_data.flatten()

            std_data = np.std(np.array(data["data"]), axis=0)
            flatten_std = std_data.flatten()

            concat_mean_std = np.concatenate((flatten_mean,flatten_std))
            appended_x = np.append(concat_mean_std,data["mean_evec"])
            
            # print(data["cla_label"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    # print(combined_x[0].shape)
    return combined_x, combined_y

def combine_data_rf_with_mean(all_data):
    #legacy
    print("combine_data_rf_with_mean")
    combined_x = []
    combined_y = []

    for key, item in all_data.items():
        for data in item:
            # print("here")
            # print(data["data"].shape)
            flatten_x = np.array(data["data"]).flatten()
            appended_x = np.append(flatten_x,data["mean_evec"])
            # print(flettened_x.shape)
            # print(data["cla_label"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y


def clean_and_combine_data(all_data):

    cleaned_data = []

    for key, items in all_data.items():
        for idx in range(0,len(items),100):
            data_item = np.array(items[idx:idx+100])
            data_cla_label = 1 if float(data_item[0][1])>0 else 0
            data_reg_label = float(data_item[0][1])
            data_item = data_item[:,3:]
            data_item = data_item.astype(np.float)
            cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"reg_label":data_reg_label})

    return cleaned_data




def random_split_data(combined_data, train_p, valid_p):

    num_training_data = int(len(combined_data)*train_p)
    num_validation_data = int(len(combined_data)*valid_p)
    num_testing_data = int(len(combined_data) - num_training_data - num_validation_data)

    #create a list of index using len(cleaned_data)
    index_list = np.arange(0,len(combined_data),1)
    # print(index_list)
    #shuffle the list
    np.random.shuffle(index_list)
    # print(index_list)

    training_data_index = index_list[:num_training_data]
    validation_data_index  = index_list[num_training_data:num_training_data+num_validation_data]
    testing_data_index = index_list[num_training_data+num_validation_data:]

    training_data = []
    validation_data = []
    testing_data = []

    #create training data
    for idx in training_data_index:
        training_data.append(combined_data[idx])
    #create training data
    for idx in validation_data_index:
        validation_data.append(combined_data[idx])
    #create training data
    for idx in testing_data_index:
        testing_data.append(combined_data[idx])

    print("{} training data, {} validation data, {} testing data".\
        format(len(training_data), len(validation_data),len(testing_data)))

    return training_data, validation_data, testing_data



class abDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self,data_dict):
        super().__init__()

        self.data = []
        self.cla_labels = []
        self.reg_labels= []

        for item in data_dict:
            #print(item["data"])
            data = torch.from_numpy(item["data"]).float()
            data = torch.transpose(data, 0, 1) #(6,100)
            #print(data.shape)
            self.data.append(data)
            self.cla_labels.append(torch.tensor(item["cla_label"]))
            self.reg_labels.append(torch.tensor(item["reg_label"])*100)

        return 

    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        __getitem__  return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {     
            "input": self.data[idx],
            "cla_labels": self.cla_labels[idx],   
            "reg_labels": self.reg_labels[idx],
        }
       
        return item


class abDataset_with_mean(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self,data_dict):
        super().__init__()

        self.data = []
        self.cla_labels = []
        self.reg_labels= []
        self.mean_evec = []

        for item in data_dict:
            #print(item["data"])
            data = torch.from_numpy(item["data"]).float()
            data = torch.transpose(data, 0, 1) #(6,100)
            #print(data.shape)
            self.data.append(data)
            self.cla_labels.append(torch.tensor(item["cla_label"]))
            self.reg_labels.append(torch.tensor(item["reg_label"])*100)
            self.mean_evec.append(torch.tensor(item["mean_evec"]))

        return 

    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        __getitem__  return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {     
            "input": self.data[idx],
            "cla_labels": self.cla_labels[idx],   
            "reg_labels": self.reg_labels[idx],
            "mean_evec": self.mean_evec[idx]
        }
       
        return item



def clean_data_with_mean_and_region(data,corre,mean_evec,region):
    cleaned_data = []
    #calculate the correlation

    for idx in range(0,len(data),100):
        mean_id = int(idx/100)
        mean_value = mean_evec[mean_id]
        region_value = region[mean_id]
        if mean_value == "nan":
            print("find nan")
            print(mean_value)
            continue
        else:
            mean_value = float(mean_value)
            region_value = float(region_value)
        data_item = np.array(data[idx:idx+100])
        chrom = data_item[0][0].split("_")[0]
        start_pos = data_item[0][0].split("_")[1]
        end_pos = data_item[-1][0].split("_")[2]
        
        evec = data_item[0][1]
        data_cla_label = 1 if float(evec)>0 else 0
        data_reg_label = float(evec)
        data_item = data_item[:,3:]
        data_item = data_item.astype(np.float)

        cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"mean_evec":mean_value,"region":region_value,\
            "reg_label":data_reg_label,"chrom":chrom,"start_pos":float(start_pos),"end_pos":float(end_pos)})

    return cleaned_data

def clean_data_for_cell_specific(data,corre,mean_evec,region):
    cleaned_data = []
    #calculate the correlation

    for idx in range(0,len(data),100):
        mean_id = int(idx/100)
        mean_value = mean_evec[mean_id]
        region_value = region[mean_id]
        if mean_value == "nan":
            print("find nan")
            print(mean_value)
            continue
        else:
            mean_value = float(mean_value)
            region_value = float(region_value)
        
        data_item = np.array(data[idx:idx+100])
        chrom = data_item[0][0].split("_")[0]
        start_pos = data_item[0][0].split("_")[1]
        end_pos = data_item[-1][0].split("_")[2]
        
        evec = data_item[0][1]
        data_cla_label = 1 if float(evec)>0 else 0
        data_reg_label = float(evec)
        data_item = data_item[:,3:]
        data_item = data_item.astype(np.float)

        if region_value == 0 and data_cla_label == 1:
            cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"mean_evec":mean_value,"region":region_value,\
            "reg_label":data_reg_label,"chrom":chrom,"start_pos":float(start_pos),"end_pos":float(end_pos)})

        elif region_value == 5 and data_cla_label == 0:
            cleaned_data.append({"data":data_item,"cla_label":data_cla_label,"mean_evec":mean_value,"region":region_value,\
            "reg_label":data_reg_label,"chrom":int(chrom[3:]),"start_pos":float(start_pos),"end_pos":float(end_pos)})

    return cleaned_data



class abDataset_with_mean_and_region(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self,data_dict):
        super().__init__()

        self.data = []
        self.cla_labels = []
        self.reg_labels= []
        self.mean_evec = []
        self.start_pos = []
        self.end_pos = []
        self.regions = []
        self.chrom = []

        for item in data_dict:
            #print(item["data"])
            data = torch.from_numpy(item["data"]).float()
            data = torch.transpose(data, 0, 1) #(6,100)
            #print(data.shape)
            self.data.append(data)
            self.cla_labels.append(torch.tensor(item["cla_label"]))
            self.reg_labels.append(torch.tensor(item["reg_label"])*100)
            self.mean_evec.append(torch.tensor(item["mean_evec"]))
            self.start_pos.append(torch.tensor(item["start_pos"]))
            self.end_pos.append(torch.tensor(item["end_pos"]))
            self.regions.append(torch.tensor(item["region"]))
            # self.chrom.append(torch.tensor(item["chrom"]))
        return 

    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        __getitem__  return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {     
            "input": self.data[idx],
            "cla_labels": self.cla_labels[idx],   
            "reg_labels": self.reg_labels[idx],
            "mean_evec": self.mean_evec[idx],
            "start_pos":self.start_pos[idx],
            "end_pos":self.end_pos[idx],
            "region":self.regions[idx],
            # "chrom":self.chrom[idx]
        }
       
        return item