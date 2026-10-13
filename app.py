"""Shadowboard server. It only fetches YouTube videos; reading the board happens in the
browser (static/engine.js), so the server needs next to no CPU.

POST /api/videos           {"url": "<youtube link>"}  -> {"video_id", status...}; starts the download
GET  /api/videos/{id}                                 -> status (downloading|ready|error), progress, error
GET  /api/videos/{id}/file                            -> the video (with range requests, so the browser can seek)
GET  /api/videos/{id}/results                         -> a finished reading a browser saved earlier
PUT  /api/videos/{id}/results                         -> save one, so the next viewer can skip the reading
"""
import json, os, re, shutil, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA = os.environ.get('SHADOWBOARD_DATA', os.path.join(os.path.dirname(__file__), 'data'))
MAX_SECONDS = float(os.environ.get('SHADOWBOARD_MAX_SECONDS', 3 * 3600))
MAX_RESULTS_MB = 8
os.makedirs(os.path.join(DATA, 'videos'), exist_ok=True)
os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)

app = FastAPI(title='Shadowboard')
videos = {}  # video id -> {'status', 'download', 'error'}
lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=int(os.environ.get('SHADOWBOARD_WORKERS', 2)))

YT_ID = re.compile(r'(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})')
VID = re.compile(r'^[A-Za-z0-9_-]{11}$')


def video_path(vid):
    return os.path.join(DATA, 'videos', vid + '.mp4')


def results_path(vid):
    return os.path.join(DATA, 'results', f'yt_{vid}.v4.json')  # same format the server used to write


def download(vid):
    import yt_dlp
    state = videos[vid]
    path = video_path(vid)

    def hook(d):
        if d.get('status') == 'downloading' and d.get('total_bytes'):
            state['download'] = d.get('downloaded_bytes', 0) / d['total_bytes']

    opts = {
        # a small video-only stream is plenty for reading a board; H.264 first, which every
        # browser can decode
        'format': 'bv*[height<=480][vcodec^=avc1]/bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/'
                  'bv*[height<=480]/b[height<=480]/worst',
        'outtmpl': os.path.join(DATA, 'videos', vid + '.part-dl.%(ext)s'),
        'quiet': True, 'noprogress': True, 'noplaylist': True,
        'progress_hooks': [hook],
        'match_filter': lambda info, *a, **k: 'too long' if (info.get('duration') or 0) > MAX_SECONDS else None,
    }
    if os.environ.get('YTDLP_COOKIES'):
        opts['cookiefile'] = os.environ['YTDLP_COOKIES']
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info('https://www.youtube.com/watch?v=' + vid, download=True)
            got = ydl.prepare_filename(info)
        if not os.path.exists(got):
            raise RuntimeError('download_failed')
        # YouTube serves fragmented MP4; a plain copy into an ordinary MP4 (no re-encoding)
        # seeks reliably in every browser
        if shutil.which('ffmpeg') and subprocess.run(
                ['ffmpeg', '-v', 'error', '-y', '-i', got, '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart',
                 '-f', 'mp4', path + '.tmp'], capture_output=True).returncode == 0:
            os.remove(got)
            os.replace(path + '.tmp', path)
        else:
            os.replace(got, path)
        state.update(status='ready', download=1.0)
    except Exception as e:  # report, don't crash the worker
        msg = str(e)
        if 'too long' in msg:
            msg = 'too_long'
        elif 'Sign in' in msg or 'bot' in msg:
            msg = 'youtube_blocked'
        else:
            msg = 'download_failed'
        state.update(status='error', error=msg)


def ensure(vid):
    """The video's state, starting its download unless it is on disk or on its way."""
    with lock:
        state = videos.get(vid)
        if os.path.exists(video_path(vid)):
            state = videos[vid] = dict(status='ready', download=1.0, error=None)
        elif not state or state['status'] == 'error':
            state = videos[vid] = dict(status='downloading', download=0.0, error=None)
            worker.submit(download, vid)
        return state


def describe(vid, state):
    return dict(video_id=vid, results=os.path.exists(results_path(vid)), **state)


class VideoIn(BaseModel):
    url: str


@app.post('/api/videos')
def create(body: VideoIn):
    m = YT_ID.search(body.url)
    if not m:
        raise HTTPException(400, 'Not a YouTube link')
    vid = m.group(1)
    return describe(vid, ensure(vid))


def known(vid):
    if not VID.match(vid):
        raise HTTPException(404, 'Unknown video')
    return vid


@app.get('/api/videos/{vid}')
def status(vid: str):
    known(vid)
    if os.path.exists(video_path(vid)):
        return describe(vid, ensure(vid))
    state = videos.get(vid)
    if not state:
        raise HTTPException(404, 'Unknown video')
    return describe(vid, state)


@app.get('/api/videos/{vid}/file')
def file(vid: str):
    path = video_path(known(vid))
    if not os.path.exists(path):
        raise HTTPException(409, 'Video not downloaded yet')
    return FileResponse(path, media_type='video/mp4')


@app.get('/api/videos/{vid}/results')
def get_results(vid: str):
    path = results_path(known(vid))
    if not os.path.exists(path):
        raise HTTPException(404, 'No results saved')
    return FileResponse(path, media_type='application/json')


class Results(BaseModel):
    events: list[dict]
    marks: list[dict]
    region: list[float] = Field(min_length=4, max_length=4)
    width: int
    height: int
    duration: float
    white_bottom: Optional[bool] = None


@app.put('/api/videos/{vid}/results')
async def put_results(vid: str, request: Request):
    known(vid)
    body = await request.body()
    if len(body) > MAX_RESULTS_MB * 1e6:
        raise HTTPException(413, 'Too large')
    try:
        res = Results.model_validate_json(body)
    except ValueError:
        raise HTTPException(400, 'Not a reading of a video')
    tmp = results_path(vid) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res.model_dump(), f)
    os.replace(tmp, results_path(vid))
    return {'ok': True}


app.mount('/', StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static'), html=True), name='static')
