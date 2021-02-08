import os
import os.path as osp

from ..stream.bodypose import BodyposeStream
from ..stream.video import VideoStream
from ..stream.depth import DepthStream

class MultiStreamVideo:
    """An media container contains mutliple streams

    Arguments:
        - path (string): the directory path to the media container

    NOTE:
        An msv container needs to contain following streams (some are optional):
            - Video stream
            - Depth stream (optional)
            - Bodypose stream (optional)
            - Feature stream (optional)
    """
    STREAMS = {
        'video': { 'suffix': 'mp4', 'type': VideoStream, 'optional': False, 'importable': False, 'exportable': False },
        'depth': { 'suffix': 'mp4', 'type': DepthStream, 'optional': True, 'importable': False, 'exportable': False },
        'bodyposes': { 'suffix': 'pkl', 'type': BodyposeStream, 'optional': True, 'importable': True,'exportable': True },
    }
    def __init__(self, path):
        self.path = path
        self.streams = {}

        # Load streaming data
        fnames = os.listdir(path)
        prefixs = [ f.split(".")[0] for f in fnames ]
        suffixs = [ f.split(".")[1] for f in fnames ]
        for stream_name, config in MultiStreamVideo.STREAMS.items():
            # No stream found
            if stream_name not in prefixs:
                self.streams[stream_name] = None
                if not config['optional']:
                    msg = f"Stream {stream_name} must include in self.__class__.__name__"
                    raise RuntimeError(msg)
                continue
            # Instantiate stream
            stream_cls = config['type']
            stream_src = "{}.{}".format(stream_name, config['suffix'])
            stream_src = osp.join(path, stream_src)
            self.streams[stream_name] = stream_cls(src=stream_src)

        # Check all streaming soucrces are in the same length
        lengths = [ len(stream)
                    for stream in self.streams.values()
                    if stream is not None ]
        assert (sum(lengths)//len(lengths)) == lengths[0]

    def __str__(self):
        content = f"{self.__class__.__name__}: {self.path}"
        for stream_name, config in MultiStreamVideo.STREAMS.items():
            stream = self.streams[stream_name]
            if stream != None:
                content += f"\n\t- Stream['{stream_name}']: available"
            else:
                content += f"\n\t- Stream['{stream_name}']: unavailable"
        return content

    def __len__(self):
        return len(self.streams['video'])

    @property
    def fps(self):
        return self.streams['video'].fps

    @property
    def width(self):
        return self.streams['video'].width

    @property
    def height(self):
        return self.streams['video'].height

    def seek(self, idx):
        for stream in self.streams.values():
            if stream is not None:
                stream.seek(idx)

    def read(self):
        frames = {}
        for stream_name, stream in self.streams.items():
            if stream is None:
                frames[stream_name] = None
                continue
            frame = stream.read()
            frames[stream_name] = frame
        return frames

    def close(self):
        for stream in self.streams.values():
            if stream is not None:
                stream.close()

    def import_stream(self, streams):
        for stream_name, src in streams.items():
            stream_cls = MultiStreamVideo.STREAMS[stream_name]['type']
            self.streams[stream_name] = stream_cls(src=src)

            fname = "{}.{}".format(stream_name, MultiStreamVideo.STREAMS[stream_name]['suffix'])
            self.streams[stream_name].src = osp.join(self.path, fname)

    def export(self):
        for stream_name, config in MultiStreamVideo.STREAMS.items():
            if config['exportable'] and self.streams[stream_name] is not None:
                self.streams[stream_name].save()
