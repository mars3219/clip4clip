from enum import Enum
import numpy as np
from .base import BaseConnecto

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# gst client
import time
import threading
from queue import Queue


class StreamMode(Enum):
    INIT_STREAM = 1
    SETUP_STREAM = 1
    READ_STREAM = 2


class StreamCommands(Enum):
    FRAME = 1
    ERROR = 2
    HEARTBEAT = 3
    RESOLUTION = 4
    STOP = 5

@CONNECTOR.register_module()
class GstStream(BaseConnector):
    def __init__(self, cfg):
        super().__init__()

        self.ch_id = cfg.ch_id
        self.streamLink = cfg.rtsp_url
        self.stop = threading.Event()
        self.outQueue = Queue(maxsize=50)
        self.saveQueue = Queue(maxsize=0)
        self.framerate = 30
        self.currentState = StreamMode.INIT_STREAM
        self.pipeline = None
        self.source = None
        self.decode = None
        self.convert = None
        self.sink = None
        self.image_arr = None
        self.newImage = False
        self.frame1 = None
        self.frame2 = None
        self.num_unexpected_tot = 400
        self.unexpected_cnt = 0
        self.state = False
        self.img_size = 1280
        self.imgs = np.zeros((720, 1280, 3), np.uint8)
        self.stride = 32
        self.auto = True
        self.fps = 25
        self.tbs = 10
        self.is_first = True
        self.connect_t = 0
        self.bus = None
        self.message = None


    def gst_to_opencv(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()

        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
             caps.get_structure(0).get_value('width'), 3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        arr = self.gst_to_opencv(sample)
        self.image_arr = arr
        self.newImage = True
        return Gst.FlowReturn.OK

    def gst_pipeline(self, streamLink, ch_id, framerate, new_buffer):
        # Create the empty pipeline
        pipeline = Gst.parse_launch(
            'rtspsrc name=m_rtspsrc ! rtph264depay name=m_rtph264depay ! avdec_h264 name=m_avdech264 ! videoconvert name=m_videoconvert ! videorate name=m_videorate ! appsink name=m_appsink')
        # pipeline = Gst.parse_launch(
        #     f'rtspsrc location="{streamLink}" latency=0 ! '
        #     # + "queue max-size-buffers=1 ! "
        #     + "decodebin ! "
        #     + "videoconvert ! "
        #     + "videoscale ! "
        #     + "video/x-raw,format=RGB,width=640,height=360 ! "
        #     # + "video/x-raw,format=RGB ! "
        #     + "appsink name=m_appsink emit-signals=true max-buffers=1 drop=true"
        # )

        # source params
        source = pipeline.get_by_name('m_rtspsrc')
        source.set_property('latency', 0)
        source.set_property('location', streamLink)
        # self.source.set_property('protocols', 'tcp')
        source.set_property('protocols', 'udp')
        source.set_property('retry', 500)
        # self.source.set_property('timeout', 50)
        # self.source.set_property('tcp-timeout', 5000000)
        source.set_property('drop-on-latency', 'true')

        # decode params
        decode = pipeline.get_by_name('m_avdech264')
        decode.set_property('max-threads', 2)
        decode.set_property('output-corrupt', 'false')

        # convert params
        convert = pipeline.get_by_name('m_videoconvert')

        # framerate parameters
        framerate_ctr = pipeline.get_by_name('m_videorate')
        framerate_ctr.set_property('max-rate', framerate/1)
        framerate_ctr.set_property('drop-only', 'true')

        # sink params
        sink = pipeline.get_by_name('m_appsink')
        sink.set_property('max-lateness', 500000000)
        sink.set_property('max-buffers', 5)
        sink.set_property('drop', 'true')
        sink.set_property('emit-signals', True)

        caps = Gst.caps_from_string(
            'video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}')
        sink.set_property('caps', caps)

        # nhs
        if not source or not sink or not pipeline or not decode or not convert:
            # LOGGER.error(f"{self.ch_id} Not all elements could be created.")
            print(f"{ch_id} Not all elements could be created.")
            self.stop.set()

        sink.connect("new-sample", new_buffer, sink)
        return pipeline
    
    def rebuild_pipeline(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        del self.pipeline
        self.pipeline = None

        self.pipeline = self.gst_pipeline(
            self.streamLink, self.ch_id, self.framerate, self.new_buffer)
        self.pipeline.set_state(Gst.State.PLAYING)

        # Wait until error or EOS
        self.bus = self.pipeline.get_bus()
        self.message = self.bus.timed_pop_filtered(10000, Gst.MessageType.ANY)

    def gst_reconnect(self):
        # 재접속 횟수만큼 시도 후 접속 실패 시 break 종료
        cnt_reconnect_attempts = 0
        reconnect_result = False
        for i in range(30):
            cnt_reconnect_attempts += 1
            print(f"Attempting to reconnect: {cnt_reconnect_attempts}")
            self.rebuild_pipeline()
            old_state, current_state, pending_state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            if current_state == Gst.State.PLAYING:
                reconnect_result = True
                cnt_reconnect_attempts = 0
                break
            time.sleep(1)
        print(f"{self.ch_id}: Reconnecttion succeess")
        
        return reconnect_result

    def producer(self):
        self.pipeline = self.gst_pipeline(
            self.streamLink, self.ch_id, self.framerate, self.new_buffer)
        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        self.Gst_Playing_T = time.time()

        # nhs
        if ret == Gst.StateChangeReturn.FAILURE:
            print(f"{self.ch_id} Unable to set the pipeline to the playing state.")
            # LOGGER.error(f"{self.ch_id} Unable to set the pipeline to the playing state.")
            self.stop.set()

        # Wait until error or EOS
        self.bus = self.pipeline.get_bus()

        st = time.time()
        fst = time.time()
        frame_cnt = 0
        second = 1

        while True:

            dt = time.time()

            if self.stop.is_set():

                print("Stopped CCTV Stream by main process")
                break

            self.message = self.bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if self.image_arr is not None and self.newImage is True:
                frame_cnt += 1
                lt = time.time()
                if int(lt-fst) == second:
                    self.fps = int(frame_cnt/second)
                    self.fps = int(15) if self.fps < 10 else self.fps
                    frame_cnt = 0
                    fst = time.time()
                    self.fps = self.fps if self.fps <= self.framerate else self.framerate

                if int(dt-st) <= self.tbs:  # previous time 저장 시간 조건 충족 전에는 지속적으로 큐에 저장함
                    self.saveQueue.put_nowait(self.image_arr)
                else:
                    self.saveQueue.get()    # previous time 저장 시간 경과 후 프레임 선입선출 시작
                    self.saveQueue.put_nowait(self.image_arr)

                if not self.outQueue.full():
                    self.outQueue.put(
                        (StreamCommands.FRAME, self.image_arr), block=False)

                self.image_arr = None
                self.unexpected_cnt = 1

            # nhs
            if self.message:
                if self.message.type == Gst.MessageType.ERROR:
                    err, debug = self.message.parse_error()
                    err_msg = "Error received from element %s: %s" % (
                        self.message.src.get_name(), err)
                        
                    reconnect_result = self.gst_reconnect()
                    if not reconnect_result:
                        print(f"Max reconnect attempts reached for {self.ch_id}. Stopping the stream.")
                        break

                # GST EOS 메세지 수신 시 소켓 전송
                elif self.message.type == Gst.MessageType.EOS:
                    print(f"{self.ch_id}: Received End-of-Stream signal.")

                    reconnect_result = self.gst_reconnect()
                    if not reconnect_result:
                        print(f"Max reconnect attempts reached for {self.ch_id}. Stopping the stream.")
                        break

                elif self.message.type == Gst.MessageType.STATE_CHANGED:
                    if isinstance(self.message.src, Gst.Pipeline):
                        old_state, new_state, pending_state = self.message.parse_state_changed()

                else:
                    self.unexpected_cnt = self.unexpected_cnt + 1
                    if self.unexpected_cnt == self.num_unexpected_tot:
                        print(
                            "{self.ch_id}: Terminate Stream with Unexpected Error")
                        break
                    
            self.fps = int(15) if self.fps < 10 else self.fps

        print(f"{self.ch_id}: stopped.")
        self.stop.set()
        self.pipeline.set_state(Gst.State.NULL)

    def consumer(self):

        while not self.stop.is_set():

            if not self.outQueue.empty() or not self.stop.is_set():
                cmd, img = self.outQueue.get()
                # print(f'outQueue size for consumer: {self.outQueue.qsize()}')
                if cmd == StreamCommands.FRAME:
                    if img is not None:
                        self.imgs = img
                time.sleep(1 / self.fps)
            else:
                break

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        return img0

    def __len__(self):
        # 1E12 frames = 32 streams at 30 FPS for 30 years
        return len(self.sources)
