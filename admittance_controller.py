#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from admittance_control.msg import IKmsg
from trac_ik_python.trac_ik import IK
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped
from admittance_control.srv import SetInt, SetIntResponse
import tf
import tf2_ros
import tf2_geometry_msgs
import json
import atexit
import time
import rospkg
import os


"""
Alternative admtitance_controller node. 
- Uses state-space formulation of the admittance controller to compute a full "dx" signal from current state "x" and current control input "u".
- Publishes 


Note that the lower-level admittance controller class features static methods for math-intensive sections of code for potential compatibility with Numba's jit compiling for fast computation.
"""


class PositionCommandPublisherNode:
    # Uses the Joint Position Controller. Publishes to the topic '/PositionController/command'
    def __init__(self, pub_topic='PositionController/command', top_type=Float64MultiArray):
        self.pub_topic = pub_topic
        self.top_type = top_type
        self.pub = rospy.Publisher(self.pub_topic, self.top_type, queue_size=10)
        self.msg = self.top_type()
        rospy.init_node('Command_Publisher', anonymous=False)

    def publish_position_handler(self, joint_angles):
        self.msg = self.top_type()
        self.msg.data = np.array(joint_angles, dtype=np.float64)
        self.pub.publish(self.msg)

class DesiredStatePublisherNode:
    # 
    def __init__(self, pose_pub_topic="ee_pose", vel_pub_topic="ee_vel", acc_pub_topic = "ee_acc", pose_top_type=PoseStamped, diff_top_type=TwistStamped):
        self.pose_top_type = pose_top_type
        self.diff_top_type = diff_top_type
        self.pose_pub = rospy.Publisher(pose_pub_topic, self.pose_top_type, queue_size=1)
        self.vel_pub = rospy.Publisher(vel_pub_topic, self.diff_top_type, queue_size=1)
        self.acc_pub = rospy.Publisher(acc_pub_topic, self.diff_top_type, queue_size=1)
        self.pose_msg = self.pose_top_type()
        self.vel_msg = self.diff_top_type()
        self.acc_msg = self.diff_top_type()
        #rospy.init_node('Command_Publisher', anonymous=False)

    def publishPose(self, full_pose):
        self.pose_msg = self.pose_top_type()
        self.pose_msg.header.stamp = rospy.Time.now()

        self.pose_msg.pose.position.x = full_pose[0]
        self.pose_msg.pose.position.y = full_pose[1]
        self.pose_msg.pose.position.z = full_pose[2]
        self.pose_msg.pose.orientation.x = full_pose[3]
        self.pose_msg.pose.orientation.y = full_pose[4]
        self.pose_msg.pose.orientation.z = full_pose[5]
        self.pose_msg.pose.orientation.w = full_pose[6]
        self.pose_msg.header.frame_id = "ee"

        self.pose_pub.publish(self.pose_msg)

    def publishTwists(self, desired_full_state):
        
        self.vel_msg = self.diff_top_type()
        self.acc_msg = self.diff_top_type()

        self.vel_msg.header.stamp = rospy.Time.now()
        self.vel_msg.twist.linear.x = desired_full_state[2]
        self.vel_msg.twist.linear.z = desired_full_state[3]
        self.vel_msg.header.frame_id = "ee"

        self.acc_msg.header.stamp = rospy.Time.now()
        self.acc_msg.twist.linear.x = desired_full_state[4]
        self.acc_msg.twist.linear.z = desired_full_state[5]
        self.acc_msg.header.frame_id = "ee"

        self.vel_pub.publish(self.vel_msg)
        self.acc_pub.publish(self.acc_msg)

class AdmittanceController(object):
    # Admittance controller class
    def __init__(self, I=10*np.eye(2), B=12*np.eye(2), K=1*np.eye(2), dim=2, pos_eq=None):
        self.I = I
        self.B = B
        self.K_active = K.copy()
        self.K_inactive = np.zeros_like(K)
        self.K = self.K_inactive.copy()

        self.dim = dim
        self.user_intent = 0.0

        if pos_eq is None:
            self.pos_eq = np.zeros((self.dim, 1))
            self.curr_pos = np.zeros((self.dim, 1))
            self.curr_vel = np.zeros((self.dim, 1))
        else:
            self.pos_eq = np.array(pos_eq).reshape((dim, 1))
            self.curr_pos = np.array(pos_eq).reshape((dim, 1))
            self.curr_vel = np.zeros((self.dim, 1))

    ### OLD
    def get_pos(self, ts, F):
        F = np.array(F).reshape((self.dim, 1))
        acc = np.linalg.inv(self.I).dot(F - self.B.dot(self.curr_vel) - self.K.dot(self.curr_pos - self.pos_eq))
        set_vel = self.curr_vel + (acc * ts)
        set_pos = self.curr_pos + (set_vel * ts)
        self.curr_vel = set_vel
        # NOTE: curr_pos is not updated by the controller. This needs to be updated manually.
        return set_pos

    ### OLD
    def get_vel(self, ts, F):
        F = np.array(F).reshape((self.dim, 1))
        acc = np.linalg.inv(self.I).dot(F - self.B.dot(self.curr_vel) - self.K.dot(self.curr_pos - self.pos_eq))
        set_vel = self.curr_vel + (acc * ts)
        # NOTE: curr_vel is not updated by the controller. This needs to be updated manually.
        return set_vel

    # functional static method for computing next state [pos, vel] and differential of next state [dpos, dvel] = [vel, acc]
    @staticmethod
    def stepAdmittanceControlStateSpace(x, dt, I, B, K, x_eq, F, verbose = False):
        # time : float
        # I, B, K : numpy array (square and invertable matrices). Should have dimension of (n x n).
        # x_t, x (t-1), x(t-2) : vector of task space. Either pose, position, or orientation with velocities. Should have dim (2n) with n being dimension of position/angle
        # force_applied : genralized applied force/wrench term. Could be torque or force or both -> implimentation depedent.
        # output: next state, x_kp1 = [x_kp1, dx_kp1] with kp1 meaning "k plus 1" (also written x_next in the code for clarity) with current state x_k = [x_k, dx_k]

        # expect x to be a 2nx1, with n being the dimension of the position/angle and the other n being the associated velocities
        # computing dimensions for the matrices
        x_dim = int(np.shape(x)[0])
        x_pos_dim = int(x_dim / 2)

        # preallocating size for big matrices
        A_dyn = np.zeros([x_dim, x_dim])
        B_dyn = np.zeros([x_dim, x_dim])
        F_dyn = np.zeros([x_dim, x_pos_dim])

        # precomputing I_inv so we don't need to compute repeatedly. Would be better if passed to function and not recomputed once per call.
        I_inv = np.linalg.inv(I)

        # filling in the big A dynamics matrix for dx = Ax + Bu
        # A_dyn[0:x_pos_dim, 0:x_pos_dim] = np.zeros([x_pos_dim, x_pos_dim]) redundant but left as a comment for clarity
        A_dyn[0:x_pos_dim, x_pos_dim:x_dim] = np.eye(x_pos_dim,x_pos_dim)
        A_dyn[x_pos_dim:x_dim, 0:x_pos_dim] = np.dot(-I_inv, K)
        A_dyn[x_pos_dim:x_dim, x_pos_dim:x_dim] = np.dot(-I_inv, B)

        B_dyn[x_pos_dim:x_dim, 0:x_pos_dim] = np.dot(I_inv, K)
        B_dyn[x_pos_dim:x_dim, x_pos_dim:x_dim] = np.dot(I_inv, B)

        F_dyn[x_pos_dim:x_dim, 0:x_pos_dim] = I_inv

        
        # computing differential state change
        #dx = np.matmul(A_dyn, x) + np.matmul(B_dyn, x_eq) + np.matmul(F_dyn, F)
        
        # make sure the F shape is right
        #F = np.reshape(F, (3,1))
        
        dx = np.dot(A_dyn, x) + np.dot(B_dyn, x_eq) + np.dot(F_dyn, F)

        # numerically integrating to get next state
        x_next = x + dt * dx

        if verbose:
            print("A_dyn: ", A_dyn)
            print("B_dyn: ", B_dyn)
            print("F_dyn: ", F_dyn)
            print("x: ", x)
            print("u (x_eq): ", x_eq)
            print("w (torque): ", F)
            print("dx: ", dx)
            print("x_next: ", x_next)

        return (x_next, dx)

    # function static method for implimenting an expoential filter for smoothing time-series vectors
    @staticmethod
    def expFiter(self, next_vec, current_vec, alpha):
        return ((alpha * next_vec) + ((1 - alpha) * current_vec))



    
class IIWAAdmittanceControllerClient(AdmittanceController):
    # Admittance Controller class client. Inherits from the generic admittance controller class.
    def __init__(self, robot_interacter, centre_pose, **kw):
        super(IIWAAdmittanceControllerClient, self).__init__(pos_eq=[centre_pose.position.x, centre_pose.position.z], **kw)
        self.robot_interacter = robot_interacter
        self.hz = 1000
        self.cycle_time = 1/self.hz
        # self.rate = rospy.Rate(self.hz)
        self.centre_pose = centre_pose
        self.force_from_sub = np.zeros(3)
        self.torque_from_sub = np.zeros(3)
        self.force_filtered = np.zeros(3)
        self.joint_bounds = np.array([[-170, 170],
                                      [-120, 120],
                                      [-170, 170],
                                      [-120, 120],
                                      [-170, 170],
                                      [-120, 120],
                                      [-175, 175]]) * (np.pi / 180)
        self.vel_bound = 6.0
        self.spline_path = None
        self.spline_time = 0.0
        self.spline_flag = False
        self.counter = 0

        self.use_variable_admittance_control = False

        # Force filter parameters:
        cutoff = 10.0
        rc = cutoff * 2 * np.pi
        self.alpha_f = rc / (rc + self.hz)

        # Position filter parameters:
        cutoff = 20.0
        rc = cutoff * 2 * np.pi
        self.alpha_p = rc / (rc + self.hz)

        # Velocity filter parameters:
        cutoff = 200.0
        rc = cutoff * 2 * np.pi
        self.alpha_v = rc / (rc + self.hz)

        self.desired_pose = [0.0 for i in range(0, 7)] # [x, y, z, rx, ry, rz, rw]
        self.desired_full_state = np.zeros(shape=(self.dim * 3, 1))

        rospy.Subscriber('/sensor_values', WrenchStamped, self.force_callback)
        rospy.Subscriber('/ee_pose_eq', PoseStamped, self.equilibriumPoseCallback)
        self.x_eq = np.zeros((4,1))

        self.set_admittance_controller_behavior_service = rospy.Service("/admit/set_admittance_controller_behavior", SetInt, self.setAdmittanceControllerBehavior)

        self.admit_control_type = 0
        self.admit_control_type_dict = {0: "Idle", 1: "Stiffness Off, Static Controller", 2: "Stiffness On, Static Controller", 3: "Stiffness On, Variable Controller"}

        self.bflag_running = False
        self.bflag_use_variable_controller = False

    def setAdmittanceControllerBehavior(self, req):
        res = SetIntResponse()
        if (req.data not in self.admit_control_type_dict.keys()):
            res.success = False
            res.message = "Invalid type! Got " + str(req.data) + ", expected one of " + str(self.admit_control_type_dict.keys()) + " for corresponding states " + str(self.admit_control_type_dict.values)
        else:
            b_behavior_changed = self.changeAdmittanceControllerBehavior(req.data)
            res.success = b_behavior_changed
            if (b_behavior_changed):
                res.message = "Set admittance controller behavior to type: " + str(self.admit_control_type_dict[self.admit_control_type]) + "!!!"
            else:
                res.message = "!!!Error!!! Internal state switching function returned false!!!"
        return res

    def changeAdmittanceControllerBehavior(self, target_state_enum):
        current_state_enum = self.admit_control_type # if need need to change things conditioned on the current state

        if (self.admit_control_type_dict[target_state_enum] == "Idle"): # switching to idle state
            # turn off running flag
            self.bflag_running = False 
            # turn off stiffness
            self.K = self.K_inactive.copy()
            
            # turn off use_variable_admittance flag
            self.bflag_use_variable_controller = False

        elif (self.admit_control_type_dict[target_state_enum] == "Stiffness Off, Static Controller"): # switching to no stiffness but active static admittance controller state
            # turn off running flag
            self.bflag_running = False 
          
            # turn off stiffness
            self.K = self.K_inactive.copy()

            # get new initial pose
            self.updateInitialPosVel()

            # turn off use_variable_admittance flag
            self.bflag_use_variable_controller = False

            # turn on running flag
            self.bflag_running = True

        elif (self.admit_control_type_dict[target_state_enum] == "Stiffness On, Static Controller"): # switching to no stiffness but active static admittance controller state
            # turn off running flag
            self.bflag_running = False 
          
            # turn on stiffness
            self.K = self.K_active.copy()

            # get new initial pose
            self.updateInitialPosVel()

            # turn off use_variable_admittance flag
            self.bflag_use_variable_controller = False

            # turn on running flag
            self.bflag_running = True

        elif (self.admit_control_type_dict[target_state_enum] == "Stiffness On, Variable Controller"): # switching to no stiffness but active static admittance controller state
            # turn off running flag
            self.bflag_running = False 
          
            # turn on stiffness
            self.K = self.K_active.copy()

            # get new initial pose
            self.updateInitialPosVel()

            # turn on use_variable_admittance flag
            self.bflag_use_variable_controller = True

            # turn on running flag
            self.bflag_running = True

        else:
            return False

        self.admit_control_type = target_state_enum
        return True

    def updateInitialPosVel(self):
        pos_vec = self.robot_interacter.getEEPositionVector()
        self.curr_pos = np.array([pos_vec[0], pos_vec[2]])
        self.curr_vel = np.array([0.0, 0.0])
        return

    def equilibriumPoseCallback(self, msg):
        self.pos_eq = np.array([msg.pose.position.x, msg.pose.position.z])
        self.x_eq = np.array([msg.pose.position.x, msg.pose.position.z, 0.0, 0.0]) # assumes zero velocity equilibrium, not true really

    def admittance_control(self, pub_obj, ik_obj, boundary, ts):
        self.force_filter(self.force_from_sub)
        F = self.force_filtered
        F = robot_interact.transform_lite("ft2base", F)

        # Update current pose (multiple methods):
        # # Option 1: No feedback from robot (do this after computing next_pose)
        # # Option 2: With active current pose subscriber
        # self.curr_pos[0, 0] = robot_interact.curr_j6_pose[0]
        # self.curr_pos[1, 0] = robot_interact.curr_j6_pose[2]

        next_pos = self.get_vel_and_filter(ts, [F[0], F[2]])
        next_pos = self.filter_pose(next_pos)

        next_pose = self.fix_pose(next_pos)
        next_pose = self.check_bounds(boundary, next_pose)  # Function call to check if end pose is within bounds

        # Get IK: From Custom IK subscriber-publisher setup
        custom_ik_node.make_ik_request(next_pose[:3], self.counter)
        # Wait till we get the matching tag for the request
        while self.counter < custom_ik_node.curr_tag:
            pass
        next_joint_angles = custom_ik_node.next_joint_angles

        self.counter += 1
        if next_joint_angles is not None:
            next_joint_angles, j_bound_flag = self.check_joint_bounds(next_joint_angles, ik_obj.seed_state, ts)
            # If not using FK server/node, update current pose if within bounds and the current joint config makes sense
            # Otherwise, stay at the same joint config, do not update current pos, do not send position command.
            if j_bound_flag:
                # Update current pose:
                # Option 1: No feedback from robot
                self.curr_pos = np.array([next_pose[0], next_pose[2]]).reshape((2, 1))

                pub_obj.publish_position_handler(np.round(next_joint_angles, 4))
        else:
            rospy.loginfo("Next joint is None!!")

    # wrapper method for collecting the correct inputs to compute the next desired/target workspace state
    def computeAdmittanceControllerDesiredState(self):
        # gets the admittance controller state in state-space form
        x = np.concatenate([self.curr_pos, self.curr_vel], axis=0)
        
        # gets filtered force values from the subscriber
        self.force_filter(self.force_from_sub)
        F = self.force_filtered
        F = robot_interact.transform_lite("ft2base", F)
        F_state = np.array([F[0], F[2]]) # state is on xz-plane for kuka reaching tasks

        if self.use_variable_admittance_control:
            self.updateVariableStiffness()

        # add subscriber for reference position
        #x_eq = np.zeros((4, 1)) # should be 2 times the state dimension in size
        # computes next state and next differential of the state
        next_state, next_dstate = self.stepAdmittanceControlStateSpace(x, self.dt, self.I, self.B, self.K, self.x_eq, F_state, verbose = False)
        

        # unwraps the pos, vel, and acc values
        next_pos = next_state[0:self.dim, 0]
        next_vel = next_dstate[0:self.dim, 0]
        next_acc = next_dstate[self.dim:(2*self.dim), 0]
        self.user_intent = np.inner(next_vel, next_acc)

        # if we assume it goes straight to the next pos/vel, update current to next for next iteration
        self.curr_pos = next_pos.copy()
        self.curr_vel = next_vel.copy()

        # creates pose [pos, quat] pair for publishing
        next_pose = self.fix_pose(next_pos) # RETURNS A LIST

        # rearranges admittance controller output for passing to history handler
        next_full_state = next_pos.tolist() + next_vel.tolist() + next_acc.tolist() # list concatenation

        self.desired_pose = next_pose
        self.desired_full_state = next_full_state
        return

    def updateVariableStiffness(self):
        return # placeholder

    def get_vel_and_filter(self, ts, F):
        set_vel = self.get_vel(ts, F)
        set_vel = (self.alpha_v * set_vel) + ((1 - self.alpha_v) * self.curr_vel)
        set_pos = self.curr_pos + (set_vel * ts)
        self.curr_vel = set_vel
        return set_pos

    def fix_pose(self, next_pos):
        return [next_pos[0], self.centre_pose.position.y, next_pos[1],
                self.centre_pose.orientation.x, self.centre_pose.orientation.y,
                self.centre_pose.orientation.z, self.centre_pose.orientation.w]

    def filter_pose(self, next_pos):
        return ((self.alpha_p * self.curr_pos) + ((1 - self.alpha_p) * next_pos))



    def check_bounds(self, boundary, gazebo_pose):
        # Function to check if the world pose of the end effector is within  the required bounds
        if gazebo_pose[0] > boundary[0][0]:
            gazebo_pose[0] = boundary[0][0]
        elif gazebo_pose[0] < boundary[0][1]:
            gazebo_pose[0] = boundary[0][1]
        
        if gazebo_pose[1] > boundary[1][0]:
            gazebo_pose[1] = boundary[1][0]
        elif gazebo_pose[1] < boundary[1][1]:
            gazebo_pose[1] = boundary[1][1]
            
        if gazebo_pose[2] > boundary[2][0]:
            gazebo_pose[2] = boundary[2][0]
        elif gazebo_pose[2] < boundary[2][1]:
            gazebo_pose[2] = boundary[2][1]
        return gazebo_pose

    def fixDesiredPoseBounds(self, boundary):
        # Function to check if the world pose of the end effector is within  the required bounds
        fixed_pose = self.desired_pose
        if fixed_pose[0] > boundary[0][0]:
            fixed_pose[0] = boundary[0][0]
        elif fixed_pose[0] < boundary[0][1]:
            fixed_pose[0] = boundary[0][1]
        
        if fixed_pose[1] > boundary[1][0]:
            fixed_pose[1] = boundary[1][0]
        elif fixed_pose[1] < boundary[1][1]:
            fixed_pose[1] = boundary[1][1]
            
        if fixed_pose[2] > boundary[2][0]:
            fixed_pose[2] = boundary[2][0]
        elif fixed_pose[2] < boundary[2][1]:
            fixed_pose[2] = boundary[2][1]
        self.desired_pose = fixed_pose
        return

    def check_joint_bounds(self, next_joint_angles, curr_joint_angles, ts):
        next_joint_angles = np.array(next_joint_angles)
        curr_joint_angles = np.array(curr_joint_angles)
        j_bound_flag = True     # Flag to check if joints are within bounds. True = Within bounds. False = outside.
        if np.any((next_joint_angles < 0.9*self.joint_bounds[:, 0]) | (next_joint_angles > 0.9*self.joint_bounds[:, 1])
                  | (np.abs(next_joint_angles - curr_joint_angles) / ts > self.vel_bound)):
            j_bound_flag = False
            next_joint_angles = curr_joint_angles   # If the bounds are not met, don't send the new angle command
        else:
            self.spline_flag = False
        return next_joint_angles, j_bound_flag

    def start(self, boundary, pub_obj, ik_obj):
        tlist = []
        i = 1
        t1t = time.time()
        t0t = t1t - 1/self.hz
        while not rospy.is_shutdown():
            tst = t1t - t0t
            tlist.append(tst)
            t0t = t1t
            # Show average loop rate every 10s
            if i % (10*self.hz) == 0:
                try:
                    rospy.loginfo("Avg freq(Hz) %s", 1/np.nanmean(tlist))
                except ZeroDivisionError:
                    rospy.loginfo("Divide by zero error!!!")
                tlist = []

            i += 1
            self.admittance_control(pub_obj, ik_obj, boundary, tst)
            while (t1t + 1/self.hz) > time.time():
                pass
            t1t = time.time()

    def runLoop(self, boundary, desired_state_pub_obj):
        time_start = time.time()
        self.waitForTime(time_start)
        while not rospy.is_shutdown():

            self.dt = time.time() - time_start
            if (self.bflag_running):
                self.computeAdmittanceControllerDesiredState() # updates desired_pose and desired_full_state
                if self.bflag_use_variable_controller:
                    self.updateVariableStiffness()
                self.fixDesiredPoseBounds(boundary)
                self.publishDesiredState(desired_state_pub_obj)
            self.waitForTime(time_start)
            time_start = time.time()

    def publishDesiredState(self, pub_obj):
        pub_obj.publishPose(self.desired_pose)
        pub_obj.publishTwists(self.desired_full_state)

    def force_callback(self, data):
        self.force_from_sub = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        self.torque_from_sub = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])

    def force_filter(self, f_new):
        f_new = self.alpha_f * self.force_filtered + (1 - self.alpha_f) * f_new
        self.force_filtered = f_new

    def waitForTime(self, start_time_point):        
        # Sleeps if faster than cycle_time
        endBeforeRest = time.time()
        elapsedTime = endBeforeRest - start_time_point

        while (elapsedTime < self.cycle_time):
            elapsedTime = time.time() - start_time_point


class RobInteract:
    # Class for other robot interactions such as transformations, workspace bounds, and current end-effector pose.
    def __init__(self):
        # Create tfBuffer and Listener for pose conversion between frames
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # Default boundary box parameters
        self.x_max = 0.1
        self.x_min = 0.1
        self.y_max = 0.1
        self.y_min = 0.1
        self.z_max = 0.1
        self.z_min = 0.1
        #time.sleep(1)

        self.curr_j6_pose = None

        # Lite transformations
        self.transform_matrices = {"ft2base": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                                   "base2ft": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])}

        rospy.Subscriber("j6_pose_custom", Float64MultiArray, self.update_curr_j6_pose)
        #rospy.Subscriber("ee_pose_custom", Float64MultiArray, self.update_curr_pose)

    # Boundary box created
    def create_bounds(self, initial_position):
        xbounds = [initial_position.position.x + self.x_max, initial_position.position.x - self.x_min]
        ybounds = [initial_position.position.y + self.y_max, initial_position.position.y - self.y_min]
        zbounds = [initial_position.position.z + self.z_max, initial_position.position.z - self.z_min]
        return xbounds, ybounds, zbounds

    def get_transformation(self, frame_from, frame_to, pose):
        # Get transformation from "frame_from" to "frame_to":
        transformation = self.tfBuffer.lookup_transform(frame_from, frame_to, rospy.Time())
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = pose[0]
        pose_stamped.pose.position.y = pose[1]
        pose_stamped.pose.position.z = pose[2]
        pose_stamped.pose.orientation.x = pose[3]
        pose_stamped.pose.orientation.y = pose[4]
        pose_stamped.pose.orientation.z = pose[5]
        pose_stamped.pose.orientation.w = pose[6]
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped_wrt_link_0 = tf2_geometry_msgs.do_transform_pose(pose_stamped, transformation)
        pose_wrt_link_0 = [pose_stamped_wrt_link_0.pose.position.x,
                           pose_stamped_wrt_link_0.pose.position.y,
                           pose_stamped_wrt_link_0.pose.position.z,
                           pose_stamped_wrt_link_0.pose.orientation.x,
                           pose_stamped_wrt_link_0.pose.orientation.y,
                           pose_stamped_wrt_link_0.pose.orientation.z,
                           pose_stamped_wrt_link_0.pose.orientation.w]
        return pose_wrt_link_0

    def get_current_ee_pose_general(self):
        # End effector pose wrt link 0 (base) pose
        #     Returns:
        #         ret_pose: Pose
        ret_pose = Pose()
        transformation = self.tfBuffer.lookup_transform("iiwa_link_0", "iiwa_link_6", rospy.Time())
        ret_pose.position = transformation.transform.translation
        ret_pose.orientation = transformation.transform.rotation
        return ret_pose

    def getEEPositionVector(self):
        # End effector pose wrt link 0 (base) pose
        #     Returns:
        #         ret_pose: Pose
        ret_pose = Pose()
        transformation = self.tfBuffer.lookup_transform("iiwa_link_0", "iiwa_link_6", rospy.Time())
        #ret_pose.position = transformation.transform.translation.x
        #ret_pose.orientation = transformation.transform.rotation
        return np.array([transformation.transform.translation.x, transformation.transform.translation.y, transformation.transform.translation.z])

    def get_current_j6_pose_custom(self):
        # End effector pose wrt link 0 (base) pose
        #     Returns:
        #         ret_pose: Pose
        ret_pose = Pose()
        ret_pose.position.x = self.curr_j6_pose[0]
        ret_pose.position.y = self.curr_j6_pose[1]
        ret_pose.position.z = self.curr_j6_pose[2]
        # TODO: This following is inaccurate because the euler angles from the custom algorithm don't match ROS
        quat = tf.transformations.quaternion_from_euler(*self.curr_j6_pose[3:])
        ret_pose.orientation.x = quat[0]
        ret_pose.orientation.y = quat[1]
        ret_pose.orientation.z = quat[2]
        ret_pose.orientation.w = quat[3]
        return ret_pose

    def transform_lite(self, transform_key, pose):
        # Lite transformations. Transform x, y, z coordinates
        pose = np.array(pose).reshape((3, 1))
        return self.transform_matrices[transform_key].dot(pose)

    def update_curr_j6_pose(self, data):
        self.curr_j6_pose = data.data

    def update_curr_pose(self, data):
        self.curr_pose = data.data


#def init_joint_angles(pub_obj):
#    joint_angles = [-np.pi/2, np.pi/2, 0, np.pi/2, 0, -np.pi/2, -55*np.pi/180]   # [x, y, z, rx, ry, rz, rw]
#    rospy.loginfo("Moving to Origin")
#    time.sleep(3)      # Wait for a while. Gazebo ignores the movement command if we don't wait here.
#    pub_obj.publish_position_handler(joint_angles)


def set_new_centre_pose(new_pose, ik_obj, pub_obj):
    next_joint_angles = ik_obj.ik_solver.get_ik(ik_obj.seed_state, *new_pose)

    if next_joint_angles is not None:
        pub_obj.publish_position_handler(next_joint_angles)
    else:
        rospy.loginfo("Could not solve IK for new pose!!")


def init_debug_dict():
    debug_dict = dict()
    keys = ["time", "loop_time_r", "loop_time_t", "curr_vel", "curr_pos", "joint_angles_raw", "joint_velocities_raw",
            "joint_velocities_bounded", "Cond1", "Cond2", "Cond3"]
    for key in keys:
        debug_dict[key] = []
    return debug_dict


def write_dict_to_file():
    try:
        fn = "debug_file.json"
        rospack = rospkg.RosPack()
        path = os.path.join(rospack.get_path('admittance_control'), "bag", fn)
        with open(path, 'w') as f:
            f.write(json.dumps(debug_dict))
    except NameError:
        rospy.loginfo("Debug dictionary not in use. Not logging.")
    else:
        rospy.loginfo("Logged dictionary to '{}'".format(path))

if __name__ == "__main__":
    nh = rospy.init_node('admittance_controller_external', anonymous=True)
    # Register log function to run on ROS exit (Use init_debug_dict to initialize)
    atexit.register(write_dict_to_file)

    pub_node = DesiredStatePublisherNode()

    #nit_joint_angles(pub_node) # what does this do?
    
    robot_interact = RobInteract()
    time.sleep(4)

    init_pose = robot_interact.get_current_j6_pose_custom()

    boundary_box = robot_interact.create_bounds(init_pose)
    rospy.loginfo("Initial pose:\n{}".format(init_pose))

    adm_controller = IIWAAdmittanceControllerClient(robot_interact, init_pose)
    #rospy.loginfo("Controller starting in 3 secs...")
    #time.sleep(3)
    #rospy.loginfo("Controller running!")
    adm_controller.runLoop(boundary_box, pub_node)

'''
if __name__ == "__main__":
    # Register log function to run on ROS exit (Use init_debug_dict to initialize)
    atexit.register(write_dict_to_file)

    pub_node = PositionCommandPublisherNode()
    ik_obj = IKSolver()
    custom_ik_node = CustomIKNode()
    init_joint_angles(pub_node)
    robot_interact = RobInteract()
    time.sleep(4)

    init_pose = robot_interact.get_current_j6_pose_custom()

    boundary_box = robot_interact.create_bounds(init_pose)
    rospy.loginfo("Initial pose:\n{}".format(init_pose))

    adm_controller = IIWAAdmittanceControllerClient(init_pose)
    rospy.loginfo("Controller starting in 3 secs...")
    time.sleep(3)
    rospy.loginfo("Controller running!")
    adm_controller.start(boundary_box, pub_node, ik_obj)
'''