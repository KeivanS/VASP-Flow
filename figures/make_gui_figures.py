#!/usr/bin/env python3
"""Screenshots of the GUI for the manual (Example 1): figures/gui_*.png.

    python3 figures/make_gui_figures.py RESULTS_FOLDER [--port 5077]

Needs Playwright with Chromium (pip install playwright && playwright install chromium) in the
Python that runs this script, and Flask in the Python of the GUI (--gui-python, default the
repository's venv/bin/python, else python3).  The GUI is started on its own port in a temporary
working directory, so no settings.json of yours is touched.  RESULTS_FOLDER is a finished
calculation (01_relax, 02_scf, ...) shown on the Results tab, e.g. the Si_example project of
Example 1.  Run from anywhere.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WIDTH = 900          # CSS px: at the full text width of the manual the GUI text is ~7 pt
SI = """Si
1.0
   0.000   2.734   2.734
   2.734   0.000   2.734
   2.734   2.734   0.000
Si
2
Direct
   0.00 0.00 0.00
   0.25 0.25 0.25
"""


def start_gui(py, port, work):
    code = ("import runpy; g = runpy.run_path(%r, run_name='vf_gui'); "
            "g['app'].run(host='127.0.0.1', port=%d, threaded=True)") % (
        os.path.join(REPO, 'vasp-gui.py'), port)
    env = dict(os.environ, VASP_PROJECTS_DIR=work)
    proc = subprocess.Popen([py, '-c', code], cwd=work, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    sys.exit('the GUI did not start')


def card(pg, title, scope='#p-setup'):
    """The card whose title starts with *title* (case-insensitive)."""
    head = pg.locator('.card-title', has_text=re.compile(r'^\s*' + re.escape(title), re.I))
    return pg.locator(scope + ' .card').filter(has=head).first


def shot(el, name):
    el.scroll_into_view_if_needed()
    el.screenshot(path=os.path.join(HERE, name))
    print('wrote figures/' + name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results')
    ap.add_argument('--port', type=int, default=5077)
    ap.add_argument('--gui-python', default=None)
    ap.add_argument('--cards', default='summary,bands,dos,elf,lobster',
                    help='Results cards to shoot (default all)')
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    py = a.gui_python or (os.path.join(REPO, 'venv', 'bin', 'python')
                          if os.path.exists(os.path.join(REPO, 'venv', 'bin', 'python')) else 'python3')
    work = tempfile.mkdtemp(prefix='vf_gui_shots_')
    gui = start_gui(py, a.port, work)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={'width': WIDTH, 'height': 1000}, device_scale_factor=2)
            pg.goto('http://127.0.0.1:%d/' % a.port)
            pg.wait_for_timeout(1500)
            pg.evaluate("document.getElementById('system-setup-body').style.display='none';"
                        "document.getElementById('system-setup-chevron').textContent='\\u25B6';"
                        "document.getElementById('alert-top').innerHTML=''")
            # Setup tab, filled in as in Example 1
            pg.fill('#project_name', 'Si_example')
            pg.fill('#description', 'Silicon: relaxation, SCF + ELF, bands, DOS, LOBSTER')
            pg.fill('#mpi_np', '8')
            pg.fill('#poscar', SI)
            pg.dispatch_event('#poscar', 'input')
            pg.dispatch_event('#poscar', 'change')
            pg.wait_for_timeout(800)
            pg.evaluate("document.getElementById('poscar').scrollTop=0")
            pg.select_option('#functional', 'PBEsol')
            pg.fill('#conv_kp', '4x4x4, 6x6x6, 8x8x8, 10x10x10, 12x12x12')
            pg.fill('#conv_encut', '250-500')
            pg.wait_for_function("document.getElementById('kpath').value !== ''", timeout=30000)
            pg.check('#t_lobster')
            pg.evaluate('addProjRow()')
            pg.wait_for_timeout(500)
            pg.evaluate('window.scrollTo(0,0)')
            shot(card(pg, 'Project'), 'gui_setup_project.png')
            shot(card(pg, 'POSCAR'), 'gui_setup_poscar.png')
            shot(card(pg, 'DFT Method'), 'gui_setup_method.png')
            shot(card(pg, 'Calculation Parameters'), 'gui_setup_conv.png')
            shot(card(pg, 'Tasks'), 'gui_setup_tasks.png')
            shot(card(pg, 'Advanced'), 'gui_setup_advanced.png')
            # Results tab of the finished calculation
            pg.fill('#open-res-path', os.path.abspath(a.results))
            pg.evaluate('openResults(false)')
            pg.wait_for_timeout(20000)
            want = a.cards.split(',')
            for key, title in (('summary', 'Calculation Summary'), ('bands', 'Band Structure'),
                               ('dos', 'Density of States'), ('elf', 'Electron Localization'),
                               ('lobster', 'LOBSTER Bonding')):
                if key in want:
                    shot(card(pg, title, '#res'), 'gui_results_%s.png' % key)
            b.close()
    finally:
        gui.terminate()


if __name__ == '__main__':
    main()
