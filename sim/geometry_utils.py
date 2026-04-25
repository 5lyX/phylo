import numpy as np

class Frame:
    def __init__(self, origin: np.array, axis_angles: np.array = np.array([0,0,0]), degrees: bool = False, quat: np.array = None):
        """
        This class represents a frame in 3D space. It is defined by an origin and an orientation.

        Args:
        - origin:  A 3D vector representing the frame's position in space.
        - axis_angles:  Euler angles (roll, pitch, yaw) in radians or degrees, which describe the frame's orientation.
        - degrees:  A boolean flag that indicates whether the axis_angles are in degrees or radians.
        - quat:  A quaternion that describes the frame's orientation. If provided, the axis_angles will be ignored.
        """
        self.origin = origin
        self.quat = quat if quat is not None else Frame.euler_to_quaternion(axis_angles, degrees)

    def rel2global(self,
        pos: np.array,
        quat: np.array = None
    ):
        """
        This method converts a point from a local (relative) frame to the global frame.

        Args:
        - pos: the position of the point in the local frame
        - quat: the orientation of the point in the local frame
        """
        pos = self.rotate(self.quat, pos) + self.origin

        if quat is not None:
            quat = Frame.quaternion_multiplication(self.quat, quat)

            return pos, quat
        
        return pos

    def global2rel(self,
        pos: np.array,
        quat: np.array = None
    ):
        """
        This method converts a point from the global frame to the local (relative) frame.

        Args:
        - pos: the position of the point in the global frame
        - quat: the orientation of the point in the global frame
        """
        # Step 1: Subtract the origin to translate the point to the local space
        pos = pos - self.origin

        # Step 2: Apply the inverse rotation (using the conjugate of the quaternion)
        pos = self.rotate(self.conjugate(self.quat), pos)

        if quat is not None:
            # Step 3: Compute the relative orientation
            quat = Frame.quaternion_multiplication(self.conjugate(self.quat), quat)

            return pos, quat
        
        return pos
    
    @staticmethod
    def rotate(q: np.array, x: np.array):
        x = (0,) + tuple(x)
        return Frame.quaternion_multiplication(Frame.quaternion_multiplication(q, x), Frame.conjugate(q))[..., 1:]

    @staticmethod
    def conjugate(q: np.array):
        q = np.array(q) # coping mechanism
        q = q * -1
        q[0] *= -1
        return q

    @staticmethod
    def quaternion_multiplication(q1: np.array, q2: np.array):
        w1, x1, y1, z1 = tuple(q1)
        w2, x2, y2, z2 = tuple(q2)
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return np.array([w, x, y, z])
    
    @staticmethod
    def euler_to_quaternion(r, degrees = False):
        if degrees: r = r / 180 * np.pi
        roll, pitch, yaw = tuple(r)
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return np.array([qw, qx, qy, qz])

    @staticmethod
    def quaternion_to_euler(q, degrees = False):
        w, x, y, z = tuple(q)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        rpy = np.array([roll, pitch, yaw])
        
        if degrees: rpy * 180 / np.pi
        return rpy
    
def test():
    x = np.array([10, 0, 0])
    frame = Frame(origin = np.zeros((3,)), axis_angles=np.array([0, 30, 0]), degrees= True)
    print(Frame.rotate(frame.quat, x))
    print(frame.quat)

if __name__ == '__main__':
    test() # Tested, it works