import cv2
import numpy as np
import datetime

class Detector():

    def __init__(self):
        self.camera = cv2.VideoCapture('./samples/768x576.mp4')
        # self.camera = cv2.VideoCapture(0)

        self.frame_shape = None

        self.f_frags = np.array([3, 5])
        self.total_f_cells = self.f_frags[0] * self.f_frags[1]

        self.p_frags = np.array([2, 7])
        self.total_p_cells = self.p_frags[0] * self.p_frags[1]

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.o_f_deg = 0.3
        self.o_p_deg = 0.8
        self.o_fp_deg = 1.

        self.memory_p = []
        self.memory_f = []
        self.mem_size = 5

        self.face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')

    @staticmethod
    def range_overlap(x1, x2, x3, x4):
        # print 'range_overlap'

        x_min = max(x1, x3)
        x_max = min(x2, x4)

        if x_min > x_max:
            return 0
        return x_max - x_min

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

    def get_pedestrians(self, frame, nw):
        # print 'get_pedestrians'

        padding = np.array([nw[1], nw[0], 0, 0])

        hog_recs, weights = self.hog.detectMultiScale(frame, winStride=(4, 4),
                                                      padding=(8, 8), scale=1.1)
        hog_recs = Detector.overlap_subtraction(hog_recs, self.o_p_deg)
        hog_recs = [hog_rec + padding for hog_rec in hog_recs if hog_rec[2] > 16 and hog_rec[3] > 16]

        return Detector.overlap_subtraction(hog_recs, self.o_p_deg)

    def get_faces(self, frame, nw):
        # print 'get_faces'

        padding = np.array([nw[1], nw[0], 0, 0])

        face_recs = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
        face_recs = Detector.overlap_subtraction(face_recs, deg=self.o_f_deg)
        face_recs = [face_rec + padding for face_rec in face_recs]

        return Detector.overlap_subtraction(face_recs, self.o_f_deg)

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

    def run(self):

        cnt = 0
        pedes = []
        faces = []

        total_cnt = 0
        cf = 0
        beg = datetime.datetime.now()
        time_f = datetime.datetime.now() - datetime.datetime.now()
        time_p = datetime.datetime.now() - datetime.datetime.now()
        time_c = datetime.datetime.now() - datetime.datetime.now()

        while True:
            cnt += 1

            ret, frame = self.camera.read()
            assert ret, 'cannot read (more)'

            frame = np.fliplr(frame).astype('uint8')
            self.frame_shape = frame.shape[:2]

            b_p = datetime.datetime.now()
            nw, sub_frame = self.get_sub_frame(frame, cnt, target='pede')
            new_pedes = self.get_pedestrians(sub_frame, nw=nw)
            time_p += datetime.datetime.now() - b_p

            b_f = datetime.datetime.now()
            nw, sub_frame = self.get_sub_frame(frame, cnt, target='face')
            new_faces = self.get_faces(sub_frame, nw=nw)
            time_f += datetime.datetime.now() - b_f

            b_c = datetime.datetime.now()
            pedes = self.consolidate(pedes, frame, objective='pedes')
            faces = self.consolidate(faces, frame, objective='faces')
            time_c += datetime.datetime.now() - b_c

            pedes = Detector.overlap_subtraction(pedes + new_pedes, self.o_p_deg)
            faces = Detector.overlap_subtraction(faces + new_faces, self.o_f_deg)
            faces = self.face_of_pede_subtraction(faces, pedes)

            total_cnt += len(pedes) + len(faces)

            '''optional'''
            new_mem_p = [(cnt, pede) for pede in pedes]
            self.memory_p = self.keep_memory(new_mem_p, self.memory_p, cnt)

            new_mem_f = [(cnt, face) for face in faces]
            self.memory_f = self.keep_memory(new_mem_f, self.memory_f, cnt)

            print 'there are {} people'.format(len(self.memory_f) + len(self.memory_p))

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

            cf +=1
            print 'frame {}, time {}, {} counted'.format(cf, datetime.datetime.now() - beg, total_cnt)
            if cf == 100:
                print 'time_p: {}, time_f: {}, time_c: {}'.format(time_p, time_f, time_c)
                break

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':

    detector = Detector()
    detector.run()
