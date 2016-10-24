import cv2
import numpy as np
import datetime

'''We use OOP approach - create a class'''
class Detector():

    def __init__(self):
        '''can use a video file or use the webcam'''
        self.camera = cv2.VideoCapture('./samples/768x576.mp4')
        # self.camera = cv2.VideoCapture(0)

        '''pre-define for future use'''
        self.frame_shape = None

        '''
        this variable define the number and the size\
        of sliding windows for face detection
        '''
        self.f_frags = np.array([3, 5])
        self.total_f_cells = self.f_frags[0] * self.f_frags[1]

        '''
        this variable define the number and the size\
        of sliding windows for pedestrian detection
        '''
        self.p_frags = np.array([2, 7])
        self.total_p_cells = self.p_frags[0] * self.p_frags[1]

        '''use HOG pedestrian detector'''
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        '''use Haar Cascades for face detection'''
        self.face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')

        '''
        these variables define the % of intersection to be discarded
        2 faces with 30% intersection, 1 will be discarded
        2 pedestrians with 80% intersection, 1 will be discarded
        1 face and 1 pedestrian with 100% intersection, the face be discarded
        '''
        self.o_f_deg = 0.3
        self.o_p_deg = 0.8
        self.o_fp_deg = 1.

        '''
        these variables are for memory
        mem_size is the maximum number of previous frames that\
        we still remember
        '''
        self.memory_p = []
        self.memory_f = []
        self.mem_size = 5

    '''
    given 2 line segments,
    the first one is defined by 2 points, x1 and x2
    the second on is defined by 2 points, x3 and x4
    this function return the length of ther intersection
    '''
    @staticmethod
    def range_overlap(x1, x2, x3, x4):
        # print 'range_overlap'

        x_min = max(x1, x3)
        x_max = min(x2, x4)

        if x_min > x_max:
            return 0
        return x_max - x_min

    '''
    given 2 rectangles and a threshold (percentage)
    this function return if their overlapping area is bigger\
    than 'percentage' percent of the smaller one
    '''
    @staticmethod
    def rec_overlap(rec1, rec2, percentage):
        # print 'rec_overlap'

        r1_x1, r1_y1, r1_w, r1_h = rec1
        r1_x2, r1_y2 = r1_x1 + r1_w, r1_y1 + r1_h
        r1_area = r1_w * r1_h

        r2_x1, r2_y1, r2_w, r2_h = rec2
        r2_x2, r2_y2 = r2_x1 + r2_w, r2_y1 + r2_h
        r2_area = r2_w * r2_h

        x_range = Detector.range_overlap(r1_x1, r1_x2, r2_x1, r2_x2)
        y_range = Detector.range_overlap(r1_y1, r1_y2, r2_y1, r2_y2)

        return 1. * x_range * y_range / min(r1_area, r2_area) >= percentage

    '''
    given a list rectangles and the threshold (deg)
    this function check if any pair of rectangles overlap >= 'deg' percent\
    and remove one accordingly
    '''
    @staticmethod
    def overlap_subtraction(recs, deg):
        # print 'overlap_substraction'

        result = []

        for rec in recs:
            for chosen in result:
                if Detector.rec_overlap(rec, chosen, deg):
                    break
            else:
                result.append(rec)

        return result

    '''
    this function check if any face (in the list of faces)\
    belongs to any pedestrian (in the list of pedestrians)
    and remove accordingly
    '''
    def face_of_pede_subtraction(self, faces, pedes):
        # print 'face_of_pede_subtraction'
        new_faces = []

        for face in faces:
            for pede in pedes:
                if Detector.rec_overlap(face, pede, self.o_fp_deg):
                    break
            else:
                new_faces.append(face)

        return new_faces

    '''
    given a frame (or sub-frame), this function detects\
    the pedestrians in that frame using HOG Detector
    nw is the north-west point of this sub-frame in the\
    whole original frame
    '''
    def get_pedestrians(self, frame, nw):
        # print 'get_pedestrians'

        padding = np.array([nw[1], nw[0], 0, 0])

        hog_recs, weights = self.hog.detectMultiScale(frame, winStride=(4, 4),
                                                      padding=(8, 8), scale=1.1)
        hog_recs = Detector.overlap_subtraction(hog_recs, self.o_p_deg)
        hog_recs = [hog_rec + padding for hog_rec in hog_recs if hog_rec[2] > 16 and hog_rec[3] > 16]

        return Detector.overlap_subtraction(hog_recs, self.o_p_deg)

    '''
    given a frame (or sub-frame), this function detects\
    the faces in that frame using Haar Cascades Detector
    nw is the north-west point of this sub-frame in the\
    whole original frame
    '''
    def get_faces(self, frame, nw):
        # print 'get_faces'

        padding = np.array([nw[1], nw[0], 0, 0])

        face_recs = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
        face_recs = Detector.overlap_subtraction(face_recs, deg=self.o_f_deg)
        face_recs = [face_rec + padding for face_rec in face_recs]

        return Detector.overlap_subtraction(face_recs, self.o_f_deg)

    '''
    depend on frame index and target (face or pedes),\
    this function will return the corresponding window
    '''
    def get_sub_frame(self, frame, cnt, target):
        # print 'get_sub_frame'

        shape = np.array(frame.shape[:2])

        if target == 'face':
            cnt %= self.total_f_cells
            idx = np.array((int(cnt / self.f_frags[1]), cnt % self.f_frags[1]))
            cell_shape = shape / (self.f_frags + 1)

        elif target == 'pede':
            cnt %= self.total_p_cells
            idx = np.array((int(cnt / self.p_frags[1]), cnt % self.p_frags[1]))
            cell_shape = shape / (self.p_frags + 1)
        else:
            print '\'target\' must be \'face\' or \'pede\''
            return None

        north_west = idx * cell_shape
        south_east = north_west + cell_shape * 2

        return north_west, frame[north_west[0]:south_east[0], north_west[1]:south_east[1], :]

    '''
    consolidate faces and pedestrians detected in the\
    last frame
    '''
    def consolidate(self, old_recs, frame, objective):
        # print 'consolidate'

        if objective == 'pedes':
            getter = self.get_pedestrians
            deg = self.o_p_deg
        elif objective == 'faces':
            getter = self.get_faces
            deg = self.o_f_deg
        else:
            print 'objective must be either \'pedes\' or \'faces\''
            return []

        '''default expand degree is 0.1'''
        result = []
        expand_idx = 0.1

        for rec in old_recs:
            x1, y1, w, h = rec
            x2, y2 = x1 + w, y1 + h

            x1 = max(0, int(x1 - expand_idx * w))
            y1 = max(0, int(y1 - expand_idx * h))
            x2 = min(self.frame_shape[1], int(x2 + expand_idx * w))
            y2 = min(self.frame_shape[0], int(y2 + expand_idx * h))

            moved = getter(frame[y1:y2, x1:x2, :], nw=(y1, x1))

            # moved = [p + np.array([x1, y1, 0, 0]) for p in moved]
            result += moved

        return Detector.overlap_subtraction(result, deg=deg)

    '''
    save the people detected for a time
    '''
    def keep_memory(self, new, old, lastest_idx):
        min_accept_idx = lastest_idx - self.mem_size

        for (idx, old_rec) in old:
            if idx < min_accept_idx:
                continue
            for (_, rec) in new:
                if Detector.rec_overlap(old_rec, rec, self.o_f_deg):
                    break
            else:
                new.append((idx, old_rec))

        return new

    '''This is the main function'''
    def run(self):
        '''count the index number of the frame'''
        cnt = 0

        '''list of pedestrians and faces detected'''
        pedes = []
        faces = []

        '''
        the number of people we have detected so far
        use for testing and measuring performance
        '''
        total_cnt = 0

        '''
        use to measure performance - time taken for each process
        time_f - time to do face detection on windows
        time_p - time to do pedestrian detection on windows
        time_c - time to consilidate detected people
        '''
        beg = datetime.datetime.now()
        time_f = datetime.datetime.now() - datetime.datetime.now()
        time_p = datetime.datetime.now() - datetime.datetime.now()
        time_c = datetime.datetime.now() - datetime.datetime.now()

        while True:
            '''increase frame index'''
            cnt += 1

            '''read next frame'''
            ret, frame = self.camera.read()
            assert ret, 'cannot read (more)'

            '''flip the frame (optional)'''
            frame = np.fliplr(frame).astype('uint8')
            '''store the resolution of the frame'''
            self.frame_shape = frame.shape[:2]

            '''
            detect pedestrian in a window
            this window is corresponding to the frame index
            '''
            b_p = datetime.datetime.now()
            nw, sub_frame = self.get_sub_frame(frame, cnt, target='pede')
            new_pedes = self.get_pedestrians(sub_frame, nw=nw)
            time_p += datetime.datetime.now() - b_p

            '''
            detect face in a window
            this window is also corresponding to the frame index
            '''
            b_f = datetime.datetime.now()
            nw, sub_frame = self.get_sub_frame(frame, cnt, target='face')
            new_faces = self.get_faces(sub_frame, nw=nw)
            time_f += datetime.datetime.now() - b_f

            '''
            consolidate the pedestrians and faces detected in the last loop
            '''
            b_c = datetime.datetime.now()
            pedes = self.consolidate(pedes, frame, objective='pedes')
            faces = self.consolidate(faces, frame, objective='faces')
            time_c += datetime.datetime.now() - b_c

            '''
            check if any pair of faces or pedestrians point to the same object
            if yes, remove 1 of them
            '''
            pedes = Detector.overlap_subtraction(pedes + new_pedes, self.o_p_deg)
            faces = Detector.overlap_subtraction(faces + new_faces, self.o_f_deg)
            faces = self.face_of_pede_subtraction(faces, pedes)

            '''count the number of people detected in all the frame so far'''
            total_cnt += len(pedes) + len(faces)

            '''
            optional - save detected people in previous frame in memory
            even if these people are not detected in the next frame, they\
            are still be counted
            '''
            new_mem_p = [(cnt, pede) for pede in pedes]
            self.memory_p = self.keep_memory(new_mem_p, self.memory_p, cnt)

            new_mem_f = [(cnt, face) for face in faces]
            self.memory_f = self.keep_memory(new_mem_f, self.memory_f, cnt)

            '''output number of people detected in this frame'''
            print 'there are {} people'.format(len(self.memory_f) + len(self.memory_p))

            '''draw rectangle around each person detected'''
            for (_, (x, y, w, h)) in self.memory_p:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

            for (_, (x, y, w, h)) in self.memory_f:
                cv2.rectangle(frame, (x - w, y - h), (x + w * 2, y + h * 5), (0, 255, 0), 4)

            # for (x, y, w, h) in pedes:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            #
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(frame, (x - w, y - h), (x + w * 2, y + h * 5), (0, 255, 0), 4)

            cv2.imshow('Human detection', frame)

            '''The below 4 lines of code is used for measuring performance only'''
            # print 'frame {}, time {}, {} counted'.format(cnt, datetime.datetime.now() - beg, total_cnt)
            # if cnt == 100:
            #     print 'time_p: {}, time_f: {}, time_c: {}'.format(time_p, time_f, time_c)
            #     break

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

'''Just declare an object and run'''
if __name__ == '__main__':
    detector = Detector()
    detector.run()
