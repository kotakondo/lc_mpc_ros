#!/usr/bin/env python3
#
# This node generates and publishes a trajectory for rovers 
# to navigate to a desired final pose

# ros imports
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from nav_msgs.msg import OccupancyGrid
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
        self.replan_lookahead_timestep = rospy.get_param("~trajectory_generator/replan_lookahead_timestep") # how many timesteps to look ahead when replanning
        self.goal_radius = rospy.get_param("~trajectory_generator/goal_radius") # radius around goal to consider goal reached

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
        
        self.Qf = np.array(rospy.get_param(f"~trajectory_generator/Qf")) # terminal cost matrix
        self.R = np.array(rospy.get_param(f"~trajectory_generator/R")) # input cost matrix
        self.terminal_cost_weight = rospy.get_param("~trajectory_generator/terminal_cost_weight") # terminal cost weight
        self.waypoints_cost_weight = rospy.get_param("~trajectory_generator/waypoints_cost_weight") # waypoints cost weight
        self.input_cost_weight = rospy.get_param("~trajectory_generator/input_cost_weight") # input cost weight
        self.travel_cost_weight = rospy.get_param("~trajectory_generator/travel_cost_weight") # travel cost weight
        self.travel_dist_cost_weight = rospy.get_param("~trajectory_generator/travel_dist_cost_weight") # travel distance cost weight
        self.input_rate_cost_weight = rospy.get_param("~trajectory_generator/input_rate_cost_weight") # input rate cost weight
        self.collision_cost_weight = rospy.get_param("~trajectory_generator/collision_cost_weight") # collision cost weight

        # Planner mode
        self.NOPLAN = 0
        self.REPLAN = 1
        self.mode = self.REPLAN

        # ROS params
        self.goal_topic = rospy.get_param("~ros/goal_topic") # topic to subscribe to for goal states
        self.costmap_topic = rospy.get_param("~ros/costmap_topic") # topic to subscribe to for costmap

        # Internal variables
        self.states = np.nan*np.ones(4)
        self.prev_states = np.nan*np.ones(4)
        self.goal_states = None
        self.u_traj = None
        self.x_traj = None
        self.t_traj = None
        self.tf = None
        self.plan_dt = None
        self.t_initialized = False
        self.start_time = None
        self.t = 0.0
        
        # Trajectory planner
        dubins = DubinsDynamics(control=CONTROL_LIN_ACC_ANG_VEL)
        map = get_map()
        self.planner = LoopClosureAwareMPC(dynamics=dubins, occupancy_map=map, num_timesteps=self.num_timesteps, 
                                           terminal_cost_weight=self.terminal_cost_weight, waypoints_cost_weight=self.waypoints_cost_weight, 
                                           input_cost_weight=self.input_cost_weight, travel_cost_weight=self.travel_cost_weight, 
                                           travel_dist_cost_weight=self.travel_dist_cost_weight, input_rate_cost_weight=self.input_rate_cost_weight, 
                                           collision_cost_weight=self.collision_cost_weight)

        # Subscribers
        # get rover pose from tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sub_pose = rospy.Timer(rospy.Duration(self.dt), self.pose_cb_tf, oneshot=False) 
        self.sub_twist = rospy.Timer(rospy.Duration(self.dt), self.twist_cb, oneshot=False)
        self.sub_goal_states = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_states_cb)
        self.sub_costmap = rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.planner.update_costmap)

        # Publishers
        # cmd_vel publisher for each rover
        self.pub_cmd_vel = rospy.Publisher(f"/cmd_vel", Twist, queue_size=1)

        # timers
        self.replaned = False
        self.replan_timer = rospy.Timer(rospy.Duration(self.dt), self.replan_cb)
        self.goal_reached_timer = rospy.Timer(rospy.Duration(1.0), self.check_plan_reached_goal)
            
    def goal_states_cb(self, msg):
        """
        Updates goal states
        """

        # if current plan has not reached goal, do not update goal states
        if self.mode != self.REPLAN and self.goal_states is not None:
            return

        print("new goal received")
        x = msg.pose.position.x 
        y = msg.pose.position.y
        v = 0.0
        theta = Rot.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_euler('xyz')[2]
        self.goal_states = np.array([x, y, v, theta])

    def cmd_vel_cb(self, event):
        """
        Loops every self.dt seconds to publish trajectory to listening robots

        Args:
            event (rospy.TimerEvent): not used
        """

        if not self.t_initialized:
            self.t = 0.0
            self.start_time = rospy.get_time()
            self.t_initialized = True

        # if trajectory has been planned, publish trajectory
        if self.u_traj is not None and self.x_traj is not None:

            # if trajectory is empty, send zero command
            if self.t >= self.tf:
                print("t >= tf")
                cmd_vel = Twist()
                cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z = 0.0, 0.0, 0.0
                self.pub_cmd_vel.publish(cmd_vel)
                return
            
            # publish cmd_vel 
            cmd_vel = Twist()
            self.t = rospy.get_time() - self.start_time
            th_cmd = np.interp(self.t, xp=self.t_traj[:-1], fp=self.x_traj[3,:-1])
            v_cmd = np.interp(self.t, xp=self.t_traj[:-1], fp=self.x_traj[2,:-1])
            th_dot_cmd = np.interp(self.t, xp=self.t_traj[:-1], fp=self.u_traj[1,:])
            cmd_vel.linear.x = v_cmd * np.cos(th_cmd)
            cmd_vel.linear.y = v_cmd * np.sin(th_cmd)
            cmd_vel.angular.z = th_dot_cmd
            self.pub_cmd_vel.publish(cmd_vel)

        return
    
    def replan_cb(self, event):

        # if all states have been received, plan trajectory, otherwise, continue waiting
        if self.all_states_received() and self.goal_states is not None and self.mode == self.REPLAN:          # and not self.replaned:

            x0 = np.array([self.states])
            xf = np.array([self.goal_states])
            
            # tf setup (initial guess)
            dist = np.linalg.norm(xf[0, 0:2] - x0[0, 0:2])
            tf_guess_factor = 2.0
            tf = abs(tf_guess_factor*dist/self.u_bounds[0,0])

            # TODO: waypoints setup (this will be the A* initial guess)
            waypoints = {}

            # solve trajectory optimization problem
            self.planner.setup_mpc_opt(x0, xf, tf, waypoints=waypoints, Qf=self.Qf, R=self.R, x_bounds=self.x_bounds, u_bounds=self.u_bounds)
            # self.planner.opti.subject_to(self.planner.tf > 1.)

            try:
                # solve optimization problem
                x_traj, u_traj, t_traj, self.cost = self.planner.solve_opt()
                print("Trajectory optimization successful")

                # initialize cm_vel_cb parameters
                self.t_initialized = False

                # store trajectory
                self.x_traj = np.array(x_traj)[0]
                self.u_traj = np.array(u_traj)[0]
                self.t_traj = np.array(t_traj)

                # store trajectory
                self.tf = self.t_traj[-1]
                self.plan_dt = self.tf / self.num_timesteps
                self.cmd_vel_timer = rospy.Timer(rospy.Duration(self.plan_dt), self.cmd_vel_cb)

            except:
                print("Trajectory optimization failed")
                return
            
            self.mode = self.NOPLAN

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
            self.states[0] = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0)).transform.translation.x
            self.states[1] = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0)).transform.translation.y
            quat = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0)).transform.rotation
            theta_unwrapped = Rot.from_quat([quat.x, quat.y, quat.w, quat.z]).as_euler('xyz')[2] + np.pi # add pi because of how theta is defined in Dubins dynamis
            self.states[3] = -((theta_unwrapped + np.pi) % (2 * np.pi) - np.pi) # wrap

            # print("states: ", self.states)
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

    def check_plan_reached_goal(self, event):
        """
        Check whether the goal has been reached
        """

        # if states, goal, or x_traj is not yet initialized, then just return
        if self.states is None or self.goal_states is None or self.x_traj is None:
            return
        
        if np.linalg.norm(self.states[:2] - self.goal_states[:2]) > self.goal_radius:
            self.mode = self.NOPLAN
            return
        
        print("goal reached!")
        self.mode = self.REPLAN
    

if __name__ == '__main__':
    rospy.init_node('trajectory_generator_node', anonymous=False)
    node = TrajectoryGeneratorNode()
    rospy.spin()
