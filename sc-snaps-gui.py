#!/usr/bin/env python3
"""
SC-Snaps GUI — browser interface for sc_snaps.x supercell snapshot generator
Run with:  python3 sc-snaps-gui.py   (or:  make snaps)
Opens http://localhost:5050
"""
import os, sys, glob, json, queue, threading
from pathlib import Path
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_EXEC = '~/BIN/sc_snaps.x'

DEFAULT_CELL = """\
1 1 1   90 90 90
 0 0.5 0.5, 0.5 0 0.5, 0.5 0.5 0
4.247
2
1 1
24.31  16.00
Mg O
  0 0 0
  0.5 0.5 0.5"""

DEFAULT_SNAPS = """\
400  Avg frequency (1/cm)
300    # temperature in K
51 1   #  of snaps needed and supercell type (default=1)"""

DEFAULT_SUPERCELL = """\
3 0 0
0 3 0
0 0 3"""

# ── job management ────────────────────────────────────────────────────────────
_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()
_job_counter = 0

def _next_job_id():
    global _job_counter
    _job_counter += 1
    return f'job_{_job_counter}'

def _run_job(job_id: str, cmd: list, cwd: str):
    q = _jobs[job_id]
    try:
        import subprocess
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in iter(proc.stdout.readline, ''):
            q.put(('out', line.rstrip()))
        proc.wait()
        q.put(('done', str(proc.returncode)))
    except Exception as exc:
        q.put(('err', str(exc)))
        q.put(('done', '1'))

# ── API ───────────────────────────────────────────────────────────────────────

@app.route('/api/run', methods=['POST'])
def api_run():
    d = request.json or {}
    workdir  = os.path.expanduser(d.get('workdir',  '').strip())
    execpath = os.path.expanduser(d.get('execpath', DEFAULT_EXEC).strip())

    if not workdir:
        return jsonify(error='Working directory is required.'), 400
    os.makedirs(workdir, exist_ok=True)

    if not os.path.isfile(execpath):
        return jsonify(error=f'Executable not found: {execpath}'), 404
    if not os.access(execpath, os.X_OK):
        return jsonify(error=f'Executable not executable (check permissions): {execpath}'), 400

    # Write input files
    Path(os.path.join(workdir, 'cell.inp')).write_text(d.get('cell', DEFAULT_CELL))
    Path(os.path.join(workdir, 'snaps.inp')).write_text(d.get('snaps', DEFAULT_SNAPS))
    Path(os.path.join(workdir, 'supercell.inp')).write_text(d.get('supercell', DEFAULT_SUPERCELL))

    job_id = _next_job_id()
    with _jobs_lock:
        _jobs[job_id] = queue.Queue()

    t = threading.Thread(target=_run_job, args=(job_id, [execpath], workdir), daemon=True)
    t.start()
    return jsonify(job_id=job_id)


@app.route('/api/stream/<job_id>')
def api_stream(job_id):
    def generate():
        with _jobs_lock:
            q = _jobs.get(job_id)
        if q is None:
            yield f'data: {json.dumps({"type":"err","line":"Job not found"})}\n\n'
            return
        while True:
            try:
                typ, line = q.get(timeout=60)
                yield f'data: {json.dumps({"type":typ,"line":line})}\n\n'
                if typ == 'done':
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/load_inputs', methods=['POST'])
def api_load_inputs():
    workdir = os.path.expanduser((request.json or {}).get('workdir', '').strip())
    result = {}
    for fname, key in [('cell.inp','cell'), ('snaps.inp','snaps'), ('supercell.inp','supercell')]:
        p = os.path.join(workdir, fname)
        result[key] = Path(p).read_text(errors='replace').strip() if os.path.isfile(p) else None
    return jsonify(result)


@app.route('/api/files', methods=['POST'])
def api_files():
    workdir = os.path.expanduser((request.json or {}).get('workdir', '').strip())
    if not os.path.isdir(workdir):
        return jsonify(files=[])
    poscars = sorted(glob.glob(os.path.join(workdir, 'poscar_*')))
    return jsonify(files=[os.path.basename(p) for p in poscars])


@app.route('/api/read_file', methods=['POST'])
def api_read_file():
    d = request.json or {}
    workdir = os.path.expanduser(d.get('workdir', '').strip())
    fname   = os.path.basename(d.get('filename', ''))   # no path traversal
    path    = os.path.join(workdir, fname)
    if not os.path.isfile(path):
        return jsonify(error='File not found'), 404
    return jsonify(content=Path(path).read_text(errors='replace'))


@app.route('/')
def index():
    return _HTML


# ── HTML / CSS / JS ──────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC-Snaps GUI</title>
<style>
:root {
  --bg:#f0f4f8; --card:#fff; --border:#e2e8f0;
  --accent:#0369a1; --accent-light:#e0f2fe;
  --text:#1e293b; --sub:#64748b;
  --code-bg:#f8fafc; --term-bg:#0f172a; --term-fg:#e2e8f0;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--text);}
header{background:var(--accent);color:#fff;padding:14px 28px;display:flex;align-items:baseline;gap:14px;}
header h1{font-size:20px;font-weight:700;letter-spacing:-.3px;}
header span{font-size:13px;opacity:.75;}
.main{max-width:1280px;margin:0 auto;padding:24px 20px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:20px;}
.card-title{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--sub);margin-bottom:14px;}
.row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;}
.f{display:flex;flex-direction:column;gap:4px;flex:1;min-width:180px;}
label{font-size:12px;font-weight:600;color:var(--sub);}
input[type=text]{border:1px solid var(--border);border-radius:6px;padding:7px 10px;font:inherit;background:var(--card);color:var(--text);width:100%;}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-light);}
textarea{border:1px solid var(--border);border-radius:6px;padding:10px 12px;
  font:13px/1.7 'Courier New',monospace;background:var(--code-bg);color:var(--text);
  resize:vertical;width:100%;tab-size:4;}
textarea:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-light);}
.files-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
@media(max-width:860px){.files-grid{grid-template-columns:1fr;}}
.btn{border:1px solid var(--border);border-radius:6px;padding:7px 16px;font:inherit;
     cursor:pointer;background:var(--card);color:var(--text);transition:background .15s;white-space:nowrap;}
.btn:hover{background:#f1f5f9;}
.btn:disabled{opacity:.45;cursor:not-allowed;}
.btn-primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600;padding:9px 22px;font-size:14.5px;}
.btn-primary:hover{background:#0284c7;}
.btn-sm{padding:4px 10px;font-size:12px;}
.btn-ghost{background:transparent;border-color:var(--border);color:var(--sub);}
.btn-ghost:hover{background:#f1f5f9;color:var(--text);}
.run-row{display:flex;gap:14px;align-items:center;margin-bottom:20px;flex-wrap:wrap;}
.run-status{font-size:13px;color:var(--sub);}
.run-status.ok{color:#15803d;font-weight:600;}
.run-status.err{color:#dc2626;font-weight:600;}
/* terminal log */
.term{background:var(--term-bg);color:var(--term-fg);font:13px/1.6 'Courier New',monospace;
      padding:16px;border-radius:8px;height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;}
.term .line-err{color:#f87171;}
.term .line-done-ok{color:#86efac;font-weight:600;}
.term .line-done-err{color:#f87171;font-weight:600;}
/* output files */
.chip-grid{display:flex;flex-wrap:wrap;gap:8px;}
.chip{background:var(--code-bg);border:1px solid var(--border);border-radius:6px;
      padding:5px 13px;font:12.5px 'Courier New',monospace;cursor:pointer;transition:all .15s;}
.chip:hover{background:var(--accent-light);border-color:var(--accent);color:var(--accent);}
.chip.active{background:var(--accent-light);border-color:var(--accent);color:var(--accent);font-weight:700;}
.viewer{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;
        padding:14px 16px;font:12.5px/1.7 'Courier New',monospace;white-space:pre;
        max-height:420px;overflow:auto;margin-top:14px;display:none;}
.badge{display:inline-block;background:var(--accent);color:#fff;border-radius:12px;
       font-size:11px;font-weight:700;padding:1px 9px;margin-left:8px;vertical-align:middle;}
.hint{font-size:11px;color:var(--sub);margin-top:4px;}
.alert{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:14px;}
.alert-err{background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;}
.alert-ok {background:#dcfce7;border:1px solid #86efac;color:#166534;}
.sep{border:none;border-top:1px solid var(--border);margin:4px 0 16px;}
</style>
</head>
<body>
<header>
  <h1>SC-Snaps GUI</h1>
  <span>Supercell snapshot generator interface</span>
</header>

<div class="main">
  <div id="alert-top"></div>

  <!-- ── Configuration ─────────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-title">Configuration</div>
    <div class="row">
      <div class="f" style="flex:3;">
        <label>Working Directory — input files are written here; snapshots appear here</label>
        <input type="text" id="workdir" placeholder="/path/to/output/directory  (new or existing)">
        <div class="hint">Type an absolute path. For a new project, just type the desired folder path — it will be created on Run. To load existing input files from an existing folder, type the path then click Load.</div>
      </div>
      <button class="btn" onclick="loadInputs()" style="margin-bottom:22px;">⬆ Load existing files</button>
    </div>
    <hr class="sep">
    <div class="row">
      <div class="f" style="flex:3;">
        <label>Executable — full path to sc_snaps.x (or its equivalent)</label>
        <input type="text" id="execpath" value="~/BIN/sc_snaps.x">
      </div>
      <div style="margin-bottom:0;padding-bottom:0;"></div>
    </div>
  </div>

  <!-- ── Input files ────────────────────────────────────────────────────── -->
  <div class="files-grid">

    <div class="card">
      <div class="card-title">cell.inp — primitive cell</div>
      <textarea id="cell" rows="12" spellcheck="false">1 1 1   90 90 90
 0 0.5 0.5, 0.5 0 0.5, 0.5 0.5 0
4.247
2
1 1
24.31  16.00
Mg O
  0 0 0
  0.5 0.5 0.5</textarea>
      <div class="hint" style="margin-top:8px;">
        Line 1: conventional cell a b c α β γ<br>
        Line 2: primitive vectors (in terms of conventional)<br>
        Line 3: lattice parameter scale (Å)<br>
        Line 4: number of atom types<br>
        Line 5: number of atoms of each type<br>
        Line 6: atomic masses<br>
        Line 7: element names<br>
        Lines 8+: reduced coordinates (conventional lattice)
      </div>
    </div>

    <div class="card">
      <div class="card-title">snaps.inp — snapshot parameters</div>
      <textarea id="snaps" rows="5" spellcheck="false">400  Avg frequency (1/cm)
300    # temperature in K
51 1   #  of snaps needed and supercell type (default=1)</textarea>
      <div class="hint" style="margin-top:8px;">
        Line 1: average phonon frequency (cm⁻¹)<br>
        Line 2: temperature (K)<br>
        Line 3: number of snapshots &amp; supercell type<br>
        <br>
        Output: <code>poscar_000</code> (unshifted) through <code>poscar_NNN</code>
      </div>
    </div>

    <div class="card">
      <div class="card-title">supercell.inp — supercell dimensions</div>
      <textarea id="supercell" rows="5" spellcheck="false">3 0 0
0 3 0
0 0 3</textarea>
      <div class="hint" style="margin-top:8px;">
        3×3 integer matrix: supercell vectors expressed<br>
        in terms of the primitive cell vectors.<br>
        A diagonal matrix <em>n n n</em> gives an n×n×n supercell.
      </div>
    </div>

  </div>

  <!-- ── Run controls ──────────────────────────────────────────────────── -->
  <div class="run-row">
    <button class="btn btn-primary" id="btn-run" onclick="runSnaps()">▶ Run sc_snaps.x</button>
    <button class="btn btn-ghost" onclick="resetDefaults()">↺ Reset defaults</button>
    <span id="run-status" class="run-status"></span>
  </div>

  <!-- ── Log ───────────────────────────────────────────────────────────── -->
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <div class="card-title" style="margin-bottom:0;">Output log</div>
      <button class="btn btn-ghost btn-sm" onclick="clearLog()">Clear</button>
    </div>
    <div class="term" id="log">Ready — configure inputs above and click Run.</div>
  </div>

  <!-- ── Generated snapshots ───────────────────────────────────────────── -->
  <div class="card" id="output-card" style="display:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <div class="card-title" style="margin-bottom:0;">
        Generated snapshots <span class="badge" id="snap-count">0</span>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="refreshFiles()">↻ Refresh</button>
    </div>
    <div class="chip-grid" id="chip-grid"></div>
    <div class="viewer" id="file-viewer"></div>
  </div>

</div><!-- .main -->

<script>
// ── defaults ────────────────────────────────────────────────────────────────
const DEFAULTS = {
  cell:
`1 1 1   90 90 90
 0 0.5 0.5, 0.5 0 0.5, 0.5 0.5 0
4.247
2
1 1
24.31  16.00
Mg O
  0 0 0
  0.5 0.5 0.5`,
  snaps:
`400  Avg frequency (1/cm)
300    # temperature in K
51 1   #  of snaps needed and supercell type (default=1)`,
  supercell:
`3 0 0
0 3 0
0 0 3`
};

// ── helpers ─────────────────────────────────────────────────────────────────
const v   = id => document.getElementById(id).value;
const el  = id => document.getElementById(id);

function alertTop(msg, type){
  el('alert-top').innerHTML = msg
    ? `<div class="alert alert-${type}">${msg}</div>` : '';
}

async function post(url, body){
  return fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
}

function resetDefaults(){
  el('cell').value      = DEFAULTS.cell;
  el('snaps').value     = DEFAULTS.snaps;
  el('supercell').value = DEFAULTS.supercell;
  alertTop('Input files reset to built-in defaults.', 'ok');
}

function clearLog(){
  el('log').innerHTML = '';
}

// ── load existing input files ────────────────────────────────────────────────
async function loadInputs(){
  const workdir = v('workdir').trim();
  if(!workdir){ alertTop('Enter a working directory path first.', 'err'); return; }
  alertTop('', '');
  const r   = await post('/api/load_inputs', {workdir});
  const d   = await r.json();
  let loaded = 0;
  if(d.cell)      { el('cell').value = d.cell;           loaded++; }
  if(d.snaps)     { el('snaps').value = d.snaps;         loaded++; }
  if(d.supercell) { el('supercell').value = d.supercell; loaded++; }
  if(loaded === 0)
    alertTop('No existing input files found in that directory — defaults are shown.', 'ok');
  else
    alertTop(`Loaded ${loaded}/3 input file(s) from: ${workdir}`, 'ok');
  await refreshFiles();
}

// ── output file list ─────────────────────────────────────────────────────────
let _activeFile = null;

async function refreshFiles(){
  const workdir = v('workdir').trim();
  if(!workdir) return;
  const r = await post('/api/files', {workdir});
  const d = await r.json();
  renderFileChips(d.files || []);
}

function renderFileChips(files){
  const card  = el('output-card');
  const grid  = el('chip-grid');
  const count = el('snap-count');
  if(files.length === 0){ card.style.display = 'none'; return; }
  card.style.display = 'block';
  count.textContent  = files.length;
  grid.innerHTML = files.map(f =>
    `<div class="chip${f===_activeFile?' active':''}" onclick="viewFile('${f}')">${f}</div>`
  ).join('');
}

async function viewFile(fname){
  // toggle off if clicking the active file
  if(_activeFile === fname){
    _activeFile = null;
    el('file-viewer').style.display = 'none';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    return;
  }
  _activeFile = fname;
  document.querySelectorAll('.chip').forEach(c =>
    c.classList.toggle('active', c.textContent === fname));

  const r = await post('/api/read_file', {workdir: v('workdir').trim(), filename: fname});
  const d = await r.json();
  const viewer = el('file-viewer');
  viewer.textContent  = d.error ? `Error: ${d.error}` : d.content;
  viewer.style.display = 'block';
  viewer.scrollTop = 0;
}

// ── run ──────────────────────────────────────────────────────────────────────
async function runSnaps(){
  const workdir = v('workdir').trim();
  if(!workdir){ alertTop('Set a working directory before running.', 'err'); return; }
  alertTop('', '');

  const btn    = el('btn-run');
  const status = el('run-status');
  btn.disabled = true;
  status.textContent = 'Running…';
  status.className   = 'run-status';

  const log = el('log');
  log.innerHTML = '';
  el('output-card').style.display = 'none';
  el('file-viewer').style.display = 'none';
  _activeFile = null;

  // Start job
  const r = await post('/api/run', {
    workdir,
    execpath:  v('execpath'),
    cell:      v('cell'),
    snaps:     v('snaps'),
    supercell: v('supercell'),
  });
  const d = await r.json();

  if(!r.ok){
    alertTop(d.error || 'Error starting job.', 'err');
    btn.disabled = false;
    status.textContent = '';
    return;
  }

  // Stream output
  const es = new EventSource(`/api/stream/${d.job_id}`);

  es.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if(msg.type === 'ping') return;

    if(msg.type === 'out' || msg.type === 'err'){
      const span = document.createElement('span');
      span.textContent = msg.line + '\n';
      if(msg.type === 'err') span.className = 'line-err';
      log.appendChild(span);
      log.scrollTop = log.scrollHeight;
    }

    if(msg.type === 'done'){
      es.close();
      const ok = msg.line === '0';
      const span = document.createElement('span');
      span.className   = ok ? 'line-done-ok' : 'line-done-err';
      span.textContent = ok
        ? '\n✓ sc_snaps.x completed successfully.\n'
        : `\n✗ sc_snaps.x exited with code ${msg.line}.\n`;
      log.appendChild(span);
      log.scrollTop = log.scrollHeight;

      btn.disabled       = false;
      status.textContent = ok ? '✓ Done' : `✗ Exit code ${msg.line}`;
      status.className   = 'run-status ' + (ok ? 'ok' : 'err');

      if(ok) await refreshFiles();
    }
  };

  es.onerror = () => {
    es.close();
    const span = document.createElement('span');
    span.className   = 'line-err';
    span.textContent = '\nSSE connection lost.\n';
    log.appendChild(span);
    btn.disabled       = false;
    status.textContent = 'Connection error';
    status.className   = 'run-status err';
  };
}
</script>
</body>
</html>
"""

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import webbrowser
    port = 5050
    threading.Timer(0.9, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    print(f'SC-Snaps GUI  →  http://localhost:{port}')
    app.run(port=port, debug=False)
