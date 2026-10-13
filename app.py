"""Shadowboard server.

POST /api/jobs            {"url": "<youtube link>"}      -> {"id": ...}
POST /api/upload          multipart "file"               -> {"id": ...}
GET  /api/jobs/{id}?since=N                              -> status, progress, board box, new events
POST /api/jobs/{id}/board {"x","y","w","h"} | {"auto": true}, optional "orientation": auto|w|b
GET  /api/jobs/{id}/frame.jpg?t=SECONDS                  -> a still frame (for fixing the board box)
GET  /api/jobs/{id}/video                                -> the uploaded file (uploads only)
"""
import hashlib, json, os, re, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline

DATA = os.environ.get('SHADOWBOARD_DATA', os.path.join(os.path.dirname(__file__), 'data'))
MAX_SECONDS = float(os.environ.get('SHADOWBOARD_MAX_SECONDS', 3 * 3600))
MAX_UPLOAD_MB = float(os.environ.get('SHADOWBOARD_MAX_UPLOAD_MB', 1024))
os.makedirs(os.path.join(DATA, 'videos'), exist_ok=True)
os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)

app = FastAPI(title='Shadowboard')
jobs = {}
lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=int(os.environ.get('SHADOWBOARD_WORKERS', 1)))

YT_ID = re.compile(r'(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})')


def new_job(**kw):
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = dict(id=jid, status='queued', progress=0.0, events=[], region=None, width=None, height=None,
                     duration=None, error=None, path=None, gen=0, white_bottom=None, orientation='auto', **kw)
    return jobs[jid]


class Cancelled(Exception):
    pass


def download(job):
    import yt_dlp
    vid = job['video_id']
    path = os.path.join(DATA, 'videos', vid + '.mp4')
    if os.path.exists(path):
        return path
    job['status'] = 'downloading'

    def hook(d):
        if d.get('status') == 'downloading' and d.get('total_bytes'):
            job['progress'] = 0.0  # processing progress starts after download
            job['download'] = d.get('downloaded_bytes', 0) / d['total_bytes']

    opts = {
        # small video-only stream is plenty for reading a board; no audio needed
        'format': 'bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480]/worst',
        'outtmpl': os.path.join(DATA, 'videos', vid + '.%(ext)s'),
        'quiet': True, 'noprogress': True, 'noplaylist': True,
        'progress_hooks': [hook],
        'match_filter': lambda info, *a, **k: 'too long' if (info.get('duration') or 0) > MAX_SECONDS else None,
    }
    if os.environ.get('YTDLP_COOKIES'):
        opts['cookiefile'] = os.environ['YTDLP_COOKIES']
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info('https://www.youtube.com/watch?v=' + vid, download=True)
        got = ydl.prepare_filename(info)
    if not os.path.exists(got):
        raise RuntimeError('download_failed')
    if got != path:
        os.replace(got, path)
    return path


def run(job, gen):
    try:
        if job['source'] == 'youtube':
            job['path'] = download(job)
        if gen != job['gen']:
            return
        cache = os.path.join(DATA, 'results', job['cache_key'] + '.json')
        if job['region'] is None and job['orientation'] == 'auto' and os.path.exists(cache):
            res = json.load(open(cache))
            job.update(res, status='done', progress=1.0)
            return
        job.update(status='finding_board', events=[], progress=0.0)

        def on_progress(p):
            if gen != job['gen']:
                raise Cancelled()
            job['status'] = 'processing'
            for k in ('region', 'width', 'height', 'duration', 'progress', 'white_bottom'):
                if k in p:
                    job[k] = p[k]
            if p.get('events'):
                with lock:
                    job['events'].extend(p['events'])

        pipeline.process(job['path'], on_progress, region=job['region'], orientation=job['orientation'])
        if gen != job['gen']:
            return
        job.update(status='done', progress=1.0)
        if job['orientation'] == 'auto':
            json.dump({k: job[k] for k in ('events', 'region', 'width', 'height', 'duration', 'white_bottom')},
                      open(cache, 'w'))
    except Cancelled:
        pass
    except Exception as e:  # report, don't crash the worker
        if gen == job['gen']:
            msg = str(e)
            if 'no_board' in msg:
                msg = 'no_board'
            elif 'too long' in msg:
                msg = 'too_long'
            elif 'Sign in' in msg or 'bot' in msg:
                msg = 'youtube_blocked'
            elif job['source'] == 'youtube' and job['status'] == 'downloading':
                msg = 'download_failed'
            job.update(status='error', error=msg)


def start(job):
    job['gen'] += 1
    worker.submit(run, job, job['gen'])


class JobIn(BaseModel):
    url: str


@app.post('/api/jobs')
def create_job(body: JobIn):
    m = YT_ID.search(body.url)
    if not m:
        raise HTTPException(400, 'Not a YouTube link')
    vid = m.group(1)
    job = new_job(source='youtube', video_id=vid, cache_key='yt_' + vid)
    start(job)
    return {'id': job['id'], 'video_id': vid}


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    h = hashlib.sha1()
    tmp = os.path.join(DATA, 'videos', 'up_' + uuid.uuid4().hex)
    size = 0
    with open(tmp, 'wb') as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1e6:
                f.close(); os.remove(tmp)
                raise HTTPException(413, 'File too large')
            h.update(chunk); f.write(chunk)
    key = 'up_' + h.hexdigest()[:16]
    path = os.path.join(DATA, 'videos', key)
    os.replace(tmp, path)
    try:
        if pipeline.probe(path)[2] > MAX_SECONDS:
            raise HTTPException(400, 'Video too long')
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, 'Not a readable video')
    job = new_job(source='upload', video_id=None, cache_key=key)
    job['path'] = path
    start(job)
    return {'id': job['id']}


def get(jid):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, 'Unknown job')
    return job


@app.get('/api/jobs/{jid}')
def status(jid: str, since: int = 0):
    job = get(jid)
    with lock:
        ev = job['events'][since:]
        total = len(job['events'])
    out = {k: job[k] for k in ('id', 'status', 'progress', 'region', 'width', 'height', 'duration', 'error',
                               'source', 'video_id', 'white_bottom', 'orientation', 'gen')}
    out.update(events=ev, total=total, download=job.get('download'))
    return out


class BoardIn(BaseModel):
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    auto: bool = False
    orientation: str | None = None


@app.post('/api/jobs/{jid}/board')
def set_board(jid: str, body: BoardIn):
    job = get(jid)
    if body.orientation in ('auto', 'w', 'b'):
        job['orientation'] = body.orientation
    if body.auto:
        job['region'] = None
    elif body.w and body.h:
        job['region'] = (body.x, body.y, body.w, body.h)
    with lock:
        job['events'] = []
    job.update(status='queued', progress=0.0, error=None)
    start(job)
    return {'ok': True, 'gen': job['gen']}


@app.get('/api/jobs/{jid}/frame.jpg')
def frame(jid: str, t: float = 0):
    job = get(jid)
    if not job.get('path') or not os.path.exists(job['path']):
        raise HTTPException(409, 'Video not ready yet')
    return Response(pipeline.frame_at(job['path'], t), media_type='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.get('/api/jobs/{jid}/video')
def video(jid: str):
    job = get(jid)
    if job['source'] != 'upload':
        raise HTTPException(404, 'Only uploaded videos are served')
    return FileResponse(job['path'], media_type='video/mp4')


app.mount('/', StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static'), html=True), name='static')
