import numpy as np
import inspect

class Env_Config:

   # simulation
   headless = True
   num_envs = 8
   sim_device = 'cuda:0'
   rl_device = 'cuda:0'
   graphics_device_id = 0
   multi_gpu = False
   seed = 1

class VecTask_cfg:

   name = 'RLDrone'
   physics_engine = 'physx'

   class env:
      numEnvs = Env_Config.num_envs
      # set to True if you use camera sensors in the environment
      enableCameraSensors = True
      numActions = 0

   class sim:
      use_gpu_pipeline = True
      up_axis = 'z'
      dt = 0.004999999888241291
      substeps = 1
      gravity =  [0.0, 0.0, -9.81]

      class physx:
         use_gpu = True

class BaseDroneEnv_Config:

   # now deprecated
   data_folder = './Datas'

   env_spacing = 5.0
   # padding (meters) to expand scene bounding box to account for object sizes
   ground_padding = 50
   env_padding = 50

   buffer_size = 100

   # camera
   class Camera:
      height = 280
      width = 280
      fov = 90
      horizontal_fov = 90
      depth_range = 50

   # action
   # roll pitch yaw
   pose_low = [-80., -80., 0.1, 0., -1/2*np.pi, 0.] 
   pose_unit = [1, 1, 1, 0., 1/12*np.pi, 1/6*np.pi]

   pose_low_idx = [0, 0, 0, 0, 0, 0]
   pose_up_idx = [160, 160, 60, 0, 12, 12]
   init_pose_idx = [80, 80, 60, 0, 12, 0]

   # state in quat
   init_state = [0, 0, 10.1, 0, 0, 0, 1]
   
class OccMapDrone_Config(BaseDroneEnv_Config):

   # point cloud
   downsample_factor = 1

   # occ_map
   occ_map_size = [5, 5, 5]

class SemanticOccDrone_Config(OccMapDrone_Config):

   # 语义图尺寸
   sem_map_size = [256, 256]

   class image_encoder:

      # semantic segmentation
      classes = [
         ['building'],
         ['people'],
         ['car']
      ]

      # 数据生成器的类别，与classes对应
      simple_city_gen_classes = [
         'building', 'ped','car' 
      ]

      # if seg with background
      with_background = True

   # 更新语义占据图时的前后命中数平滑系数
   alpha = 0.5

   # 高度图最小值，以区分未探索（0）
   min_height = 0.5

class RLDrone_Config(OccMapDrone_Config):

   # coverage threshold
   coverage_threshold = 0.99
   coverage_scale = 1000
   termination_scale = 0
   short_path_scale = 5

class RLDrone_Sem_Config(SemanticOccDrone_Config):

   # 需要重建的class的idx
   scan_classes_idx = [0]

   # RL参数
   # height_map覆盖率参数
   # height_map差值比率
   residual_ratio_threshold = 0.1
   # 当height_map达到该覆盖率时，认为任务完成
   height_coverage_threshold = 0.9
   height_coverage_scale = 1000

   # 终止条件参数
   termination_scale = 0

   # 短路径奖励
   short_path_scale = 5
   # 当超过该距离时惩罚
   short_path_threshold = 30
   # 最高惩罚步数
   max_penalty_steps = 2

class Train_Config:

   pass