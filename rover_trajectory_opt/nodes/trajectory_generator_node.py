#!/usr/bin/env python3
#
# This node generates and publishes a trajectory for rovers 
# to navigate to a desired final pose

# ros imports
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from std_srvs.srv import Trigger
from rover_trajectory_msgs.msg import RoverState

# python imports
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# trajectory optimization imports:
import lcmpc.dubins_dynamics as dubins_dynamics
from lcmpc.dubins_dynamics import DubinsDynamics, CONTROL_LIN_ACC_ANG_VEL
from lcmpc.occupancy_map import OccupancyMap
from lcmpc.loop_closure_aware_mpc import LoopClosureAwareMPC

def get_map():

    map_size_x = 30
    map_size_y = 30
    num_obstacles = 0
    start_x = 25
    start_y = 2
    goal_x = 5
    goal_y = 20

    return OccupancyMap(map_size_x, map_size_y, num_obstacles, start_x, start_y, goal_x, goal_y)

class TrajectoryGeneratorNode():

    def __init__(self):
        self.node_name = rospy.get_name()
        
        # Params
        self.rover = rospy.get_param("~trajectory_generator/rover")           # rover names
        self.dt = rospy.get_param("~trajectory_generator/dt")                   # how often to publish trajectory
        self.num_timesteps = rospy.get_param("~trajectory_generator/num_timesteps")  # number of timesteps to use in trajectory optimization
        self.goal_states = np.array(rospy.get_param(f"~trajectory_generator/goal_states")) # dictionary mapping rover names to goal states
        
        x_bounds = np.array(rospy.get_param(f"~trajectory_generator/x_bounds")) # state boundaries
        u_bounds = np.array(rospy.get_param(f"~trajectory_generator/u_bounds")) # input boundaries
        # reformatting boundaries for input into tomma
        self.x_bounds = np.zeros(x_bounds.shape) 
        self.u_bounds = np.zeros(u_bounds.shape)
        for i in range(self.x_bounds.shape[0]):
            for j in range(self.x_bounds.shape[1]):
                self.x_bounds[i,j] = float(x_bounds[i,j])
        for i in range(self.u_bounds.shape[0]):
            for j in range(self.u_bounds.shape[1]):
                self.u_bounds[i,j] = float(u_bounds[i,j])
        
        self.terminal_cost_weight = rospy.get_param("~trajectory_generator/terminal_cost_weight") # terminal cost weight
        self.waypoints_cost_weight = rospy.get_param("~trajectory_generator/waypoints_cost_weight") # waypoints cost weight
        self.input_cost_weight = rospy.get_param("~trajectory_generator/input_cost_weight") # input cost weight
        self.travel_cost_weight = rospy.get_param("~trajectory_generator/travel_cost_weight") # travel cost weight
        self.travel_dist_cost_weight = rospy.get_param("~trajectory_generator/travel_dist_cost_weight") # travel distance cost weight
        self.input_rate_cost_weight = rospy.get_param("~trajectory_generator/input_rate_cost_weight") # input rate cost weight
        self.collision_cost_weight = rospy.get_param("~trajectory_generator/collision_cost_weight") # collision cost weight

        # Internal variables
        self.states = np.nan*np.ones(4)
        self.prev_states = np.nan*np.ones(4)
        
        # Trajectory planner
        dubins = DubinsDynamics(control=CONTROL_LIN_ACC_ANG_VEL)
        map = get_map()
        self.planner = LoopClosureAwareMPC(dynamics=dubins, occupancy_map=map, num_timesteps=self.num_timesteps, 
                                           terminal_cost_weight=self.terminal_cost_weight, waypoints_cost_weight=self.waypoints_cost_weight, 
                                           input_cost_weight=self.input_cost_weight, travel_cost_weight=self.travel_cost_weight, 
                                           travel_dist_cost_weight=self.travel_dist_cost_weight, input_rate_cost_weight=self.input_rate_cost_weight, 
                                           collision_cost_weight=self.collision_cost_weight)

        # Trajectory variables
        self.traj_planned = False
        self.t = 0.0
        
        # Pub & Sub
        # get rover pose from tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sub_pose = rospy.Timer(rospy.Duration(self.dt), self.pose_cb_tf, oneshot=False) 
        self.sub_twist = rospy.Timer(rospy.Duration(self.dt), self.twist_cb, oneshot=False)
        
        # cmd_vel publisher for each rover
        self.pub_cmd_vel = rospy.Publisher(f"/cmd_vel", Twist, queue_size=1)

        # timers
        self.replaned = False
        self.replan_timer = rospy.Timer(rospy.Duration(self.dt), self.replan_cb)
            
    def cmd_vel_cb(self, event):
        """
        Loops every self.dt seconds to publish trajectory to listening robots

        Args:
            event (rospy.TimerEvent): not used
        """
        # if trajectory has been planned, publish trajectory
        if self.traj_planned:

            # if trajectory is empty, send zero command
            if self.u_traj.shape[1] == 0:
                cmd_vel = Twist()
                cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z = 0.0, 0.0, 0.0
                cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z = 0.0, 0.0, 0.0
                self.pub_cmd_vel.publish(cmd_vel)
                return
            
            # publish cmd_vel
            cmd_vel = Twist()
            
            # extract linear and angular velocities from trajectory
            x = self.x_traj[0,0] 
            y = self.x_traj[1,0]
            v = self.x_traj[2,0]
            theta = self.x_traj[3,0]
            vdot = self.u_traj[0,0]
            thetadot = self.u_traj[1,0]
            vx = v*np.cos(theta)
            vy = v*np.sin(theta)

            cmd_vel.linear.x = vx
            cmd_vel.linear.y = vy
            cmd_vel.linear.z = 0.0
            cmd_vel.angular.x = 0.0
            cmd_vel.angular.y = 0.0
            cmd_vel.angular.z = thetadot
            self.pub_cmd_vel.publish(cmd_vel)

            # remove first element of trajectory
            self.x_traj = np.delete(self.x_traj, 0, 1)
            self.u_traj = np.delete(self.u_traj, 0, 1)

        return
    
    def replan_cb(self, event):

        # if all states have been received, plan trajectory, otherwise, continue waiting
        if self.all_states_received() and not self.traj_planned:

            # plan trajectory
            # x0 and xf setup
            x0 = np.array([self.states])
            xf = np.array([self.goal_states])
            
            # tf setup (initial guess)
            dist = np.linalg.norm(xf[0, 0:2] - x0[0, 0:2])
            tf_guess_factor = 2.0
            tf = abs(tf_guess_factor*dist/self.u_bounds[0,0])

            # TODO: waypoints setup (this will be the A* initial guess)
            waypoints = {}

            # solve trajectory optimization problem
            self.planner.setup_mpc_opt(x0, xf, tf, waypoints=waypoints, x_bounds=self.x_bounds, u_bounds=self.u_bounds)
            # self.planner.opti.subject_to(self.planner.tf > 1.)
            x_traj, u_traj, t_traj, self.cost = self.planner.solve_opt()

            self.x_traj = np.array(x_traj)[0]
            self.u_traj = np.array(u_traj)[0]
            self.t_traj = np.array(t_traj)

            # store trajectory
            self.tf = self.t_traj[-1]
            self.traj_planned = True
            self.plan_dt = self.tf / self.num_timesteps
            self.cmd_vel_timer = rospy.Timer(rospy.Duration(self.plan_dt), self.cmd_vel_cb)

            self.replanned = True

        else:
            return # wait for all starting positions to be known
        
    def pose_cb_tf(self, event):
        """
        Stores the most recent pose (with theta wrapping)
        
        """

        try: # https://stackoverflow.com/questions/54596517/ros-tf-transform-cannot-find-a-frame-which-actually-exists-can-be-traced-with-r

            # for twist we need to store the previous pose
            self.prev_states = self.states[:]

            # get most recent pose
            self.states[0] = self.tf_buffer.lookup_transform("map", "odom", rospy.Time()).transform.translation.x
            self.states[1] = self.tf_buffer.lookup_transform("map", "odom", rospy.Time()).transform.translation.y
            quat = self.tf_buffer.lookup_transform("map", "odom", rospy.Time()).transform.rotation
            theta_unwrapped = Rot.from_quat([quat.x, quat.y, quat.w, quat.z]).as_euler('xyz')[2] + np.pi # add pi because of how theta is defined in Dubins dynamis
            self.states[3] = -((theta_unwrapped + np.pi) % (2 * np.pi) - np.pi) # wrap
        except:
            return

    def twist_cb(self, event):
        """
        Compute the most recent linear velcoties
        (If twist msg is available then we can just subscribe to that instead of calculating it from the pose)

        Args:
            twist_stamped (TwistStamped): rover twist
        """

        if not np.any(np.isnan(self.states[:2])): # if pos has been received

            prev_x = self.prev_states[0]
            prev_y = self.prev_states[1]

            current_x = self.states[0]
            current_y = self.states[1]

            # get most recent velocity
            self.states[2] = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)/self.dt
        
    def all_states_received(self):
        """
        Check whether each rover's state has been recieved
        """
        return False if np.any(np.isnan(self.states)) else True
    
if __name__ == '__main__':
    rospy.init_node('trajectory_generator_node', anonymous=False)
    node = TrajectoryGeneratorNode()
    rospy.spin()
