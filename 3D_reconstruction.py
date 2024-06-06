import configparser

from sgm import Parameters as SGMParameters, Paths,get_recall, compute_costs, compute_costs_by_improved_census, aggregate_costs, normalize, select_disparity, create_disparity_map
from image import Image_processing,downsample_image,create_output
from depth import depth_map

import numpy as np

import cv2
import time as t


from multiprocessing import Process, Manager

class Reconstruction:
    def __init__(self) -> None:
        # self.params = None
        pass

    def load_config(self, file_path = 'config.ini'):
        config = configparser.ConfigParser()
        config.read(file_path)
        
        config_dict = {}
        for section in config.sections():
            config_dict[section] = {}
            for key in config[section]:
                config_dict[section][key] = config[section][key]
        if config_dict['General']['sgm_version'] == 'sgm':
            self.params = SGMParameters(max_disparity=int(config_dict['SGMParams']['max_disparity']), 
                                        P1=int(config_dict['SGMParams']['p1']), 
                                        P2=int(config_dict['SGMParams']['p2']), 
                                        csize=tuple(map(int, config_dict['SGMParams']['csize'][1:len(config_dict['SGMParams']['csize'])-1].split(', '))), 
                                        bsize=tuple(map(int, config_dict['SGMParams']['bsize'][1:len(config_dict['SGMParams']['bsize'])-1].split(', '))))
        elif config_dict['General']['SGMParams'] == 'adcsgm':
            self.params = SGMParameters(max_disparity=int(config_dict['SGMParams']['max_disparity']), 
                                        P1=int(config_dict['SGMParams']['p1']), 
                                        P2=int(config_dict['SGMParams']['p2']), 
                                        csize=tuple(map(int, config_dict['SGMParams']['csize'][1:len(config_dict['SGMParams']['csize'])-1].split(', '))), 
                                        bsize=tuple(map(int, config_dict['SGMParams']['bsize'][1:len(config_dict['SGMParams']['bsize'])-1].split(', '))))
            print("good choise")
        else:
            raise AssertionError("wrong algoritm parameter")

        self.img_path_left = config_dict['General']['img_path_left']
        self.img_path_right = config_dict['General']['img_path_right']

        self.scale = float(config_dict['Preprocess']['scaling'])

        self.point_cloud_folder = config_dict['Saving']['point_cloud_folder']
        self.disparity_map_folder = config_dict['Saving']['disparity_map_folder']

        self.parallel_running = config_dict['General']['parallel_running'] == "True"

        self.save_disparity_map = config_dict['Evaluation']['save_disparity_map'] == "True"
        self.compute_metrics = config_dict['Evaluation']['compute_metrics'] == "True"
        self.gt_path = config_dict['Evaluation']['gt_path']



        print("self.parallel_running = ", self.parallel_running)
        return config_dict

    def preprocessing(self):
        self.imgL = Image_processing(self.img_path_left, self.scale)
        self.imgR = Image_processing(self.img_path_right, self.scale)

    def load_images(self):
        pass
    
    def create_disparity_map_1(self):
        Map = create_disparity_map(self.imgL,self.imgR)
        return Map

    def create_disparity_map(self, compute_costs_function, aggregate_costs_function):
            
        dawn = t.time()
        paths = Paths()

        print('\nStarting cost computation...')
        #compute_costs
        left_cost_volume, right_cost_volume = compute_costs_function(self.imgL, self.imgR, self.params, False) # _by_improved_census
        # compute_costs_by_improved_census
        print("left_cost_volume =", np.unique(left_cost_volume))
        print("right_cost_volume =", np.unique(right_cost_volume))

        
        if self.parallel_running:
            print('\nStarting paralel aggregation computation...')
            aggregation_volume = [None, None]

            manager = Manager()
            aggregation_volume = manager.dict()
            p1 = Process(target=aggregate_costs_function, args=(left_cost_volume, self.params, paths, aggregation_volume, 0))#, daemon=True)
            p2 = Process(target=aggregate_costs_function, args=(right_cost_volume, self.params, paths, aggregation_volume, 1))#, daemon=True)
            p1.start()
            p2.start()
            p2.join()
            p1.join()
            aggregation_volume = aggregation_volume.values()
            left_aggregation_volume = aggregation_volume[0]
            right_aggregation_volume = aggregation_volume[1]
        else: 
            print('\nStarting aggregation computation...')

            left_aggregation_volume = aggregate_costs_function(left_cost_volume, self.params, paths)#, aggregation_volume, 0) #aggregation_volume[0]
            # print('\nStarting right aggregation computation...')
            right_aggregation_volume = aggregate_costs_function(right_cost_volume, self.params, paths) #
            # print(left_aggregation_volume)
        # print(right_aggregation_volume)
        print('\nSelecting best disparities...')
        # print("left_aggregation_volume =", np.unique(left_aggregation_volume))
        # print("right_aggregation_volume =", np.unique(right_aggregation_volume))

        left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), self.params))
        right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), self.params))
        
        dusk = t.time()


        print('\nFin.')
        print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))

        # print("left_disparity_map =", np.unique(left_disparity_map))
        # print("right_disparity_map =", np.unique(right_disparity_map))

        if self.save_disparity_map:
            cv2.imwrite(self.disparity_map_folder + 'left_disp_map_no_post_processing.png', left_disparity_map)
            cv2.imwrite(self.disparity_map_folder + 'left_disp.png', cv2.medianBlur(left_disparity_map, self.params.bsize[0]))
        # left_disparity_map = cv2.medianBlur(left_disparity_map, self.params.bsize[0])
        # cv2.imwrite(f'left_disp', left_disparity_map)
        if self.compute_metrics:
            recall = get_recall(left_disparity_map, self.gt_path, self.params.max_disparity)
            print('\tRecall = {:.2f}%'.format(recall * 100.0))

        self.left_disparity_map = left_disparity_map
        self.right_disparity_map = right_disparity_map
        

        return left_disparity_map


    def create_depth_map(self):
        img=cv2.imread(self.img_path_left,1)
        img= downsample_image(img,1,self.scale)
        self.coordinates = depth_map(self.left_disparity_map,img)


    def save_ply(self):
        print('\n Creating the output file... \n')
        create_output(self.coordinates,self.point_cloud_folder + 'praxis.ply')


if __name__ == "__main__":
    reconstruction = Reconstruction()
    reconstruction.load_config()

    reconstruction.preprocessing()

    reconstruction.create_disparity_map(compute_costs_by_improved_census, aggregate_costs) # _by_improved_census

    reconstruction.create_depth_map()
    
    reconstruction.save_ply()
