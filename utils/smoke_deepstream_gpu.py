"""Exercise Savant's DeepStream native bindings on a GPU host."""

import importlib

import cv2
import gi
import numpy as np
import pyds
import pysavantboost

gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402


def run_pipeline(pipeline: Gst.Pipeline) -> None:
    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    message = pipeline.get_bus().timed_pop_filtered(
        15 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, 'Pipeline timed out.'
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        raise RuntimeError(f'{error}: {debug}')


def check_cuda_imports() -> None:
    for module in (
        'deepstream_nvbufsurface',
        'deepstream_encoders',
        'pygstsavantframemeta',
        'pynvbufsurfacegenerator',
    ):
        importlib.import_module(module)
    assert hasattr(cv2.savant, 'createGpuMat')
    frame = cv2.cuda.GpuMat()
    frame.upload(np.zeros((8, 8, 4), dtype=np.uint8))
    assert frame.download().shape == (8, 8, 4)


def check_rotated_crop() -> None:
    pipeline = Gst.parse_launch(
        'videotestsrc num-buffers=1 pattern=white ! '
        'video/x-raw,format=RGBA,width=64,height=64 ! '
        'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! '
        'identity name=sample ! fakesink sync=false'
    )
    frames = []
    errors = []

    def crop_frame(_pad, info):
        try:
            frame = pysavantboost.cut_rotated_bbox(
                16.0, 16.0, 16.0, 16.0, 0.0, 0, 0, hash(info.get_buffer()), 0
            )
            frames.append(np.array(frame, copy=True))
        except Exception as error:
            errors.append(error)
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name('sample').get_static_pad('src').add_probe(
        Gst.PadProbeType.BUFFER, crop_frame
    )
    run_pipeline(pipeline)
    assert not errors, errors
    assert len(frames) == 1, len(frames)
    assert frames[0].shape == (16, 16, 4), frames[0].shape
    assert np.all(frames[0] == 255), frames[0].min()


def check_preprocessing_restore() -> int:
    pipeline = Gst.parse_launch(
        'nvstreammux name=mux batch-size=1 ! '
        'identity name=sample ! nvvideoconvert ! '
        'video/x-raw,format=RGBA ! identity name=cpu ! fakesink sync=false '
        'videotestsrc num-buffers=1 pattern=smpte ! '
        'video/x-raw,format=RGBA,width=64,height=64 ! '
        'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! mux.sink_0'
    )
    processor = pysavantboost.ObjectsPreprocessing()
    observed = {}
    errors = []

    def preprocess(_pad, info):
        try:
            buffer = info.get_buffer()
            surface = pysavantboost.PyDSCudaMemory(hash(buffer), 0)
            observed['pitch'] = surface.pitch
            gpu_frame = cv2.savant.createGpuMat(
                surface.height,
                surface.width,
                cv2.CV_8UC4,
                surface.GetMapCudaPtr(),
                surface.pitch,
            )
            observed['before'] = gpu_frame.download()
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            assert batch_meta is not None
            frame_meta = pyds.NvDsFrameMeta.cast(batch_meta.frame_meta_list.data)
            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            obj_meta.class_id = 1
            obj_meta.unique_component_id = 1
            obj_meta.rect_params.left = 32.0
            obj_meta.rect_params.top = 16.0
            obj_meta.rect_params.width = 16.0
            obj_meta.rect_params.height = 16.0
            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
            processor.preprocessing('smoke', hash(buffer), 1, 1)
            observed['processed'] = gpu_frame.download()
            observed['object_position'] = (
                obj_meta.rect_params.left,
                obj_meta.rect_params.top,
            )
            processor.restore_frame(hash(buffer))
            observed['restored'] = gpu_frame.download()
            surface.UnMapCudaPtr()
        except Exception as error:
            errors.append(error)
        return Gst.PadProbeReturn.OK

    def capture(_pad, info):
        buffer = info.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        assert ok
        observed['frame'] = np.frombuffer(mapped.data, dtype=np.uint8).copy()
        buffer.unmap(mapped)
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name('sample').get_static_pad('src').add_probe(
        Gst.PadProbeType.BUFFER, preprocess
    )
    pipeline.get_by_name('cpu').get_static_pad('src').add_probe(
        Gst.PadProbeType.BUFFER, capture
    )
    run_pipeline(pipeline)
    assert not errors, errors
    assert observed['object_position'] == (0.0, 0.0), observed['object_position']
    assert observed['pitch'] >= 64 * 4, observed['pitch']
    before = observed['before']
    assert before.shape == (64, 64, 4), before.shape
    source_crop = before[16:32, 32:48]
    assert not np.array_equal(before[:16, :16], source_crop)
    assert np.array_equal(observed['processed'][:16, :16], source_crop)
    assert np.array_equal(observed['restored'], before)
    assert observed['frame'].size == 64 * 64 * 4
    return observed['pitch']


if __name__ == '__main__':
    Gst.init(None)
    check_cuda_imports()
    check_rotated_crop()
    pitch = check_preprocessing_restore()
    print(f'DeepStream GPU smoke passed (NVMM row pitch: {pitch} bytes).')
