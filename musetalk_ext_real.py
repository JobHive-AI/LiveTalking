###############################################################################
#  MuseTalk-Ext integration for LiveTalking UI
###############################################################################
from __future__ import annotations

import asyncio
import queue
import time
from pathlib import Path
import sys
from threading import Thread

import numpy as np
from av import AudioFrame

from basereal import BaseReal
from logger import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import LiveTalkingAdapter, MusetalkAdapter


_PRESET_MAP = {
    "calm": (0.1, 0.2),
    "confident": (0.2, 0.4),
    "excited": (0.6, 0.8),
}


def _resolve_avatar_dir(opt) -> str:
    if getattr(opt, "musetalk_ext_avatar_dir", None):
        return opt.musetalk_ext_avatar_dir
    root = Path(__file__).resolve().parents[1]
    return str(root / "musetalk_ext" / "avatars" / opt.avatar_id)


def _resolve_device(opt) -> str:
    if getattr(opt, "musetalk_ext_device", None):
        return opt.musetalk_ext_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class MusetalkExtReal(BaseReal):
    def __init__(self, opt, model=None, avatar=None):
        super().__init__(opt)
        self._audio_queue: queue.Queue[tuple[np.ndarray, object | None]] = queue.Queue(maxsize=3000)
        self._emotion_preset = "confident"

        video_fps = max(1, int(opt.fps / 2))
        avatar_dir = _resolve_avatar_dir(opt)
        device = _resolve_device(opt)
        logger.info("musetalk_ext avatar_dir=%s device=%s fps=%s", avatar_dir, device, video_fps)

        self._engine = MusetalkAdapter(
            avatar_dir=avatar_dir,
            device=device,
            fps=video_fps,
        )
        self.set_emotion_preset(self._emotion_preset)

    def set_emotion_preset(self, preset: str) -> None:
        preset = preset.lower()
        if preset not in _PRESET_MAP:
            logger.warning("Unknown emotion preset: %s", preset)
            return
        self._emotion_preset = preset
        valence, arousal = _PRESET_MAP[preset]
        self._engine.set_emotion(valence, arousal)

    def put_audio_frame(self, audio_chunk, datainfo: dict = {}):
        eventpoint = None
        if isinstance(datainfo, dict):
            eventpoint = datainfo.get("eventpoint")
        else:
            eventpoint = datainfo
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        try:
            self._audio_queue.put((chunk, eventpoint), timeout=0.5)
        except queue.Full:
            logger.warning("audio queue full, dropping chunk")

    def flush_talk(self):
        try:
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass
        self.tts.flush_talk()

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)
        if video_track is None:
            logger.error("LiveTalking video track missing")
            return

        video_output = LiveTalkingAdapter(video_track, loop)

        def _run():
            frame_count = 0
            last_log = time.time()
            # Push a short silence warmup so the UI shows a frame immediately.
            silence = np.zeros(self.chunk, dtype=np.float32)
            for _ in range(5):
                pcm16 = (silence * 32767).astype(np.int16)
                self._engine.push_audio(pcm16)
                for frame in self._engine.generate():
                    video_output.send(frame, time.time())
                    frame_count += 1
                time.sleep(0.02)
            while not quit_event.is_set():
                try:
                    chunk, eventpoint = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                pcm16 = np.clip(chunk, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype(np.int16)
                self._engine.push_audio(pcm16)

                if audio_track is not None:
                    frame = AudioFrame(format="s16", layout="mono", samples=pcm16.shape[0])
                    frame.planes[0].update(pcm16.tobytes())
                    frame.sample_rate = 16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((frame, eventpoint)), loop)

                for frame in self._engine.generate():
                    video_output.send(frame, time.time())
                    frame_count += 1

                if time.time() - last_log > 2.0:
                    if frame_count > 0:
                        logger.info("musetalk_ext sent %d frames", frame_count)
                        frame_count = 0
                    last_log = time.time()

                if video_track._queue.qsize() > 50:
                    time.sleep(0.01)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        while not quit_event.is_set():
            time.sleep(0.05)

        logger.info("musetalk_ext render stop")
