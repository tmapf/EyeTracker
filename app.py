import sys
import os

# Work around Qt plugin conflicts on Linux when OpenCV ships its own Qt
# plugins. We try to remove OpenCV's plugin path from Qt environment
# variables so PyQt5 loads the correct platform plugin.
try:
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)
    cv2_qt_plugins = os.path.join(cv2_dir, 'qt', 'plugins')
    # Remove cv2's plugin path from QT_PLUGIN_PATH and QT_QPA_PLATFORM_PLUGIN_PATH
    for var in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        val = os.environ.get(var)
        if val and cv2_qt_plugins in val:
            parts = [p for p in val.split(os.pathsep) if p and cv2_qt_plugins not in p]
            if parts:
                os.environ[var] = os.pathsep.join(parts)
            else:
                del os.environ[var]
except Exception:
    pass

# Hint platform based on session type when not explicitly set
if 'QT_QPA_PLATFORM' not in os.environ:
    s = os.environ.get('XDG_SESSION_TYPE', '').lower()
    if s == 'wayland':
        os.environ['QT_QPA_PLATFORM'] = 'wayland'
    else:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2  # noqa: E402 (re-import allowed)
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# Make sure the project dir is importable when running app.py directly
sys.path.insert(0, os.path.dirname(__file__) or '.')
from main import EyeTracker


def cvimg_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
    if img is None:
        return QtGui.QPixmap()

    # Ensure contiguous array for QImage
    arr = np.ascontiguousarray(img)
    h, w = arr.shape[:2]

    if arr.ndim == 2:
        qimg = QtGui.QImage(arr.data, w, h, arr.strides[0], QtGui.QImage.Format_Grayscale8)
    else:
        # convert BGR (OpenCV) to RGB
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)

    # Copy into QPixmap to avoid referencing the numpy memory after function returns
    return QtGui.QPixmap.fromImage(qimg.copy())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeTracker GUI")
        self.et = EyeTracker()
        self.et.init_blob_detector()

        self.cap = cv2.VideoCapture(0)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        grid = QtWidgets.QGridLayout(central)

        # Five preview labels
        self.lbl_whole = QtWidgets.QLabel()
        self.lbl_left = QtWidgets.QLabel()
        self.lbl_right = QtWidgets.QLabel()
        self.lbl_left_thresh = QtWidgets.QLabel()
        self.lbl_right_thresh = QtWidgets.QLabel()

        for lbl in (self.lbl_whole, self.lbl_left, self.lbl_right, self.lbl_left_thresh, self.lbl_right_thresh):
            lbl.setFixedSize(320, 240)
            lbl.setStyleSheet("background: #222;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)

        grid.addWidget(self.lbl_whole, 0, 0)
        grid.addWidget(self.lbl_left, 0, 1)
        grid.addWidget(self.lbl_right, 0, 2)
        grid.addWidget(self.lbl_left_thresh, 1, 1)
        grid.addWidget(self.lbl_right_thresh, 1, 2)

        # Sliders
        s_layout = QtWidgets.QHBoxLayout()
        self.s_l = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s_l.setRange(0, 255)
        self.s_l.setValue(self.et.l_thresh)
        self.s_r = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s_r.setRange(0, 255)
        self.s_r.setValue(self.et.r_thresh)

        self.lbl_l_val = QtWidgets.QLabel(str(self.et.l_thresh))
        self.lbl_r_val = QtWidgets.QLabel(str(self.et.r_thresh))

        s_layout.addWidget(QtWidgets.QLabel("L Threshold"))
        s_layout.addWidget(self.s_l)
        s_layout.addWidget(self.lbl_l_val)
        s_layout.addSpacing(20)
        s_layout.addWidget(QtWidgets.QLabel("R Threshold"))
        s_layout.addWidget(self.s_r)
        s_layout.addWidget(self.lbl_r_val)

        grid.addLayout(s_layout, 2, 0, 1, 3)

        # Connections
        self.s_l.valueChanged.connect(self.on_l_change)
        self.s_r.valueChanged.connect(self.on_r_change)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def on_l_change(self, v):
        self.et.l_thresh = int(v)
        self.lbl_l_val.setText(str(v))

    def on_r_change(self, v):
        self.et.r_thresh = int(v)
        self.lbl_r_val.setText(str(v))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Show whole frame
        self.lbl_whole.setPixmap(cvimg_to_qpixmap(frame))

        face = self.et.detect_face(frame)
        if face is None:
            # clear other views
            self.lbl_left.setPixmap(QtGui.QPixmap())
            self.lbl_right.setPixmap(QtGui.QPixmap())
            self.lbl_left_thresh.setPixmap(QtGui.QPixmap())
            self.lbl_right_thresh.setPixmap(QtGui.QPixmap())
            return

        left_eye, right_eye = self.et.detect_eyes(face)

        if left_eye is not None:
            left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            left_kp, left_thresh = self.et.blob_track(left_gray, self.et.l_thresh, self.et.previous_left_blob_area)
            if left_kp:
                try:
                    drawn = self.et.drawKeypoints(left_eye.copy(), left_eye.copy(), left_kp)
                    self.et.previous_left_blob_area = left_kp[0].size
                except Exception:
                    drawn = left_eye
            else:
                drawn = left_eye
            self.lbl_left.setPixmap(cvimg_to_qpixmap(drawn))
            # left threshold image is single-channel; make it 3-channel for display
            left_display = cv2.cvtColor(left_thresh, cv2.COLOR_GRAY2BGR)
            self.lbl_left_thresh.setPixmap(cvimg_to_qpixmap(left_display))
        else:
            self.lbl_left.setPixmap(QtGui.QPixmap())
            self.lbl_left_thresh.setPixmap(QtGui.QPixmap())

        if right_eye is not None:
            right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            right_kp, right_thresh = self.et.blob_track(right_gray, self.et.r_thresh, self.et.previous_right_blob_area)
            if right_kp:
                try:
                    drawn_r = self.et.drawKeypoints(right_eye.copy(), right_eye.copy(), right_kp)
                    self.et.previous_right_blob_area = right_kp[0].size
                except Exception:
                    drawn_r = right_eye
            else:
                drawn_r = right_eye
            self.lbl_right.setPixmap(cvimg_to_qpixmap(drawn_r))
            right_display = cv2.cvtColor(right_thresh, cv2.COLOR_GRAY2BGR)
            self.lbl_right_thresh.setPixmap(cvimg_to_qpixmap(right_display))
        else:
            self.lbl_right.setPixmap(QtGui.QPixmap())
            self.lbl_right_thresh.setPixmap(QtGui.QPixmap())

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
