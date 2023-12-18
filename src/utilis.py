import pickle

def save_object(file_path, object):
    with open(file_path, 'wb') as file:
        pickle.dump(object, file)