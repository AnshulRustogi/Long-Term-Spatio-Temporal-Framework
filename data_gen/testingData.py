import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from preprocess import pre_normalization

#List of training subjects for xsub benchmark
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
#List of training camera for xview benchmark
training_cameras = [2, 3]

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
toolbar_width = 30

import numpy as np
import os

#Printing the toolbar
#-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-XX-X-X-X
def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")
#-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X

def read_skeleton_filter(file):
    #opening the file as read
    with open(file, 'r') as f:
        #Creating a dictionary
        skeleton_sequence = {}
        #Storing number of frames in the file
        skeleton_sequence['numFrame'] = int(f.readline())
        #Creating a list of dictionaries to store the frame info
        skeleton_sequence['frameInfo'] = []
        
        #For each frame
        for t in range(skeleton_sequence['numFrame']):

            frame_info = {}
            #Number of bodies in the frame
            frame_info['numBody'] = int(f.readline())

            #Information for each body stored in a seperate list inside the dictionary
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                #Storing information regarding the body
                body_info_key = ['bodyID', 'clipedEdges', 'handLeftConfidence',
                                 'handLeftState', 'handRightConfidence', 'handRightState',
                                 'isResticted', 'leanX', 'leanY', 'trackingState']
                body_info = {k: float(v) for k, v in zip(body_info_key, f.readline().split())}
                #Next lines shows the number of joints in the body
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                #Now for each joint
                for v in range(body_info['numJoint']):
                    joint_info_key = ['x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                                      'orientationW', 'orientationX', 'orientationY',
                                      'orientationZ', 'trackingState']
                    joint_info = {k: float(v) for k, v in zip(joint_info_key, f.readline().split())}
                    body_info['jointInfo'].append(joint_info)
                
                frame_info['bodyInfo'].append(body_info)
            
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence

def get_nonzero_std(s): 
    index = s.sum(-1).sum(-1) != 0
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std() 
    else:
        s = 0
    return s

def read_xyz(file, max_body=4, num_joint=25):
    """
    read_skeleton_filter function retuns a dictionary with keys: numFrame, frameInfo (a list of dictionaries which stores the information for each frame). lenght of frameInfo list = numFrame
    For frameInfo list it contains dictionary with keys numBody, bodyInfo (a list of dictionaries which stores the information for each body). lenght of bodyInfo list = numBody
    For bodyInfo list it contains dictionary with keys 'bodyID', 'clipedEdges', 'handLeftConfidence',
                                 'handLeftState', 'handRightConfidence', 'handRightState',
                                 'isResticted', 'leanX', 'leanY', 'trackingState', numJoint , jointInfo (a list of dictionaries which stores the information for each joint of the body). lenght of jointInfo list = numJoints
    For jointInfo list it contains dictionary with keys, 'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                                      'orientationW', 'orientationX', 'orientationY',
                                      'orientationZ', 'trackingState'. For each body
    """
    seq_info = read_skeleton_filter(file)

    #Creating an numpy zero array of size (max_body=4, #frames, num_joint=25, 3)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))

    for n, f in enumerate(seq_info['frameInfo']):
        #n = index (0, numFrames), f = bodyInfo dictionary list
        for m, b in enumerate(f['bodyInfo']):
            #m = index (0 numBody), b = jointInfo dictionary list
            for j, v in enumerate(b['jointInfo']):
                #j = index (0, numJoints), v = v differenent dictionaries 
                if m < max_body and j < num_joint:
                    #Storing on 0,1,2,4 bodies, and 25 joints information as
                    #Data stores the xyz coordinates for each joint (j) for each body (m) in each frame(n)
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    #data shape = xyz<->3, numFrames, numJoints, numBody
    return data

def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xsub', set_name='val'):

    #Creating a list of files which incomplete samples
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []

    #Initialising two list
    sample_name = []
    sample_label = []
    #Looping through all the files in the booking
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue

        #Recognising action_class, subject_id and camera_id from filename

        #Example: S006C003P024R002A045.skeleton: action class=45, subject id=24, camera id=3
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        #Categorising the file as istraining or not
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub_full':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()
        
        if set_name == 'train_full':
            issample = istraining
        elif set_name == 'val_full':
            issample = not (istraining)
        else:
            raise ValueError()

        #If file is a sample file, then add the file to sample_name list and it's label to sample_label list
        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
    print("Label: " + str(len(sample_label)))
    #Opening the file as a write binary file, and dumping the sample name and sample label to that file.
    with open('{}/{}_label.pkl'.format(out_path, set_name), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    #Creating an numpy zero array of dimension [lenght of sample_label, 3, max_frame=300, num_joint=25, max_body_true=2]
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    #Looping through the sampl_name list
    for i, s in enumerate(sample_name):
        #--Simply printing the toolbar
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, set_name))
        #--XXXXXXXXXXXXXXXXXXXXXXXXXXX

        #data shape = xyz<->3, numFrames, numJoints, numBody
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint) #max_body_kinect=4, num_joint = 25
        #fp[file count, 3, 0:numFrames, numJoints, numBody]
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()

    fp = pre_normalization(fp)
    #Store the entire numpy array in the file in shape, [file count, 3, 0:numFrames, numJoints, numBody]
    np.save('{}/{}_data_joint_pad.npy'.format(out_path, set_name), fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='data/NTU-RGB+D/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path', default='data/NTU-RGB+D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/nturgb_d/')

    benchmark = ['xsub_full']
    set_name = ['train_full', 'val_full']
    arg = parser.parse_args()

    for b in benchmark:
        for sn in set_name:
            out_path = os.path.join(arg.out_folder, b)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, sn)
            gendata(arg.data_path, out_path, arg.ignored_sample_path, benchmark=b, set_name=sn)
