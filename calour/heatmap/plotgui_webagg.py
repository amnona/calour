# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import html
import json
import threading
from pathlib import Path
from string import Template
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import backend_webagg

from .plotgui import PlotGUI
from .._doc import ds


logger = getLogger(__name__)


@ds.with_indent(4)
class PlotGUI_WebAgg(PlotGUI):
    '''WebAgg GUI for interactive heatmap in a web browser.

    This GUI uses matplotlib's WebAgg backend to render an interactive heatmap
    canvas in the browser, and hosts a lightweight companion web page with
    live selection info (sample/feature/abundance and selected rows/columns).

    The plot interactions are handled by the base :class:`PlotGUI` callbacks:
    mouse click selection, keyboard zooming and keyboard scrolling.

    Notes
    -----
    Keyboard focus must be on the plot frame to receive shortcut keys.
    '''

    _manager_cache = {}
    _orig_get_fig_manager = None
    _side_panel_template = None

    @ds.with_indent(8)
    def __init__(self, side_panel_port=0, open_browser=True, **kwargs):
        '''Init the GUI using matplotlib WebAgg and a lightweight web server.

        Keyword Arguments
        -----------------
        %(PlotGUI.parameters)s
        side_panel_port : int, optional
            Port for the companion info page. 0 picks an ephemeral free port.
        open_browser : bool, optional
            True (default) to open the companion page automatically.
        '''
        super().__init__(**kwargs)
        self._set_figure(None, kwargs['tree_size'])
        # Start with a larger canvas so the heatmap fills the left pane.
        self.resize_figure(15, 10)

        self.side_panel_port = side_panel_port
        self.open_browser = open_browser

        self._manager = None
        self._webagg_base_url = None
        self._figure_url = None
        self._side_server = None
        self._side_server_thread = None

        self._last_info = {
            'sample_id': 'na',
            'feature_id': 'na',
            'abundance': '0',
            'selected_samples': [],
            'selected_features': [],
        }

    def __call__(self):
        '''Run the WebAgg GUI.

        Starts:
        1. Matplotlib WebAgg server for the interactive canvas.
        2. Companion info server for selection details and shortcuts.

        Then blocks while the WebAgg server event loop is running.
        '''
        super().__call__()

        if self._manager is None:
            self._attach_figure_to_webagg()
        self._install_gcf_fallback()
        self._ensure_manager_registered()

        backend_webagg.WebAggApplication.initialize()
        self._webagg_base_url = 'http://{address}:{port}{prefix}'.format(
            address=backend_webagg.WebAggApplication.address,
            port=backend_webagg.WebAggApplication.port,
            prefix=backend_webagg.WebAggApplication.url_prefix,
        )
        logger.info(
            'WebAgg manager registration: num=%s lookup_ok=%s gcf_keys=%s',
            self._manager.num,
            Gcf.get_fig_manager(self._manager.num) is not None,
            list(Gcf.figs.keys()),
        )
        self._figure_url = '{base}/{fignum}'.format(base=self._webagg_base_url, fignum=self._manager.num)
        print('Calour WebAgg figure URL: {url}'.format(url=self._figure_url))

        self._start_side_server()
        page_url = 'http://127.0.0.1:{port}/'.format(port=self._side_server.server_port)
        logger.info('Web heatmap companion page: %s', page_url)
        print('Calour WebAgg heatmap is available at: {url}'.format(url=page_url))

        if self.open_browser:
            webbrowser.open(page_url)

        # Blocks by design while WebAgg is active.
        backend_webagg.WebAggApplication.start()

    def _attach_figure_to_webagg(self):
        # Remove any existing pyplot manager for this figure (created by plt.figure()),
        # so this figure is owned by a single WebAgg manager.
        old_num = None
        for cnum, cmanager in list(Gcf.figs.items()):
            if cmanager.canvas.figure is self.figure:
                old_num = cnum
                break

        if old_num is not None:
            old_manager = Gcf.figs.pop(old_num)
            if hasattr(old_manager, '_cidgcf'):
                old_manager.canvas.mpl_disconnect(old_manager._cidgcf)

        nums = [m.num for m in Gcf.get_all_fig_managers()]
        if old_num is not None and old_num not in Gcf.figs:
            new_num = old_num
        else:
            new_num = (max(nums) + 1) if nums else 1

        self._manager = backend_webagg.new_figure_manager_given_figure(new_num, self.figure)
        if self._manager is None:
            raise RuntimeError('Failed to create WebAgg figure manager')
        self.__class__._manager_cache[self._manager.num] = self._manager

        # `new_figure_manager_given_figure` does not always register the manager in Gcf.
        # WebAgg resolves figure URLs via Gcf lookup, so we must register explicitly.
        if hasattr(Gcf, '_set_new_active_manager'):
            Gcf._set_new_active_manager(self._manager)
        else:
            Gcf.set_active(self._manager)
        self._ensure_manager_registered()
        logger.debug('Attached figure to WebAgg manager #%d', self._manager.num)

    @classmethod
    def _install_gcf_fallback(cls):
        '''Install a fallback for environments that clear Gcf between requests.'''
        if cls._orig_get_fig_manager is not None:
            return

        cls._orig_get_fig_manager = Gcf.get_fig_manager

        def _get_fig_manager_with_fallback(gcf_cls, num):
            manager = cls._orig_get_fig_manager.__func__(gcf_cls, num)
            if manager is not None:
                return manager

            manager = cls._manager_cache.get(num)
            if manager is not None:
                gcf_cls.figs[num] = manager
                gcf_cls.set_active(manager)
            return manager

        Gcf.get_fig_manager = classmethod(_get_fig_manager_with_fallback)

    def _ensure_manager_registered(self):
        '''Ensure the WebAgg manager can be resolved from Gcf by figure number.'''
        if self._manager is None:
            return

        if Gcf.get_fig_manager(self._manager.num) is not None:
            return

        # Fallback path for environments where manager registration can be lost.
        Gcf.figs[self._manager.num] = self._manager
        if hasattr(Gcf, 'set_active'):
            Gcf.set_active(self._manager)

    def _start_side_server(self):
        gui = self

        class SidePanelHandler(BaseHTTPRequestHandler):
            def _send_json(self, payload, status=200):
                body = json.dumps(payload).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, body, status=200):
                encoded = body.encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == '/api/info':
                    self._send_json(gui._collect_side_info())
                    return

                if parsed.path != '/':
                    self._send_json({'error': 'not found'}, status=404)
                    return

                self._send_html(gui._render_side_panel_html())

            def log_message(self, fmt, *args):
                # Silence per-request noise.
                return

        self._side_server = ThreadingHTTPServer(('127.0.0.1', self.side_panel_port), SidePanelHandler)
        self._side_server_thread = threading.Thread(target=self._side_server.serve_forever, daemon=True)
        self._side_server_thread.start()

    def _collect_side_info(self):
        sid, fid, abd, annt = self.get_info()

        sample_fields = [str(c) for c in self.exp.sample_metadata.columns]
        feature_fields = [str(c) for c in self.exp.feature_metadata.columns]

        sample_values = {}
        feature_values = {}

        if sid != 'na' and sid in self.exp.sample_metadata.index:
            for c in self.exp.sample_metadata.columns:
                sample_values[str(c)] = str(self.exp.sample_metadata.loc[sid, c])

        if fid != 'na' and fid in self.exp.feature_metadata.index:
            for c in self.exp.feature_metadata.columns:
                feature_values[str(c)] = str(self.exp.feature_metadata.loc[fid, c])

        self._last_info = {
            'sample_id': str(sid),
            'feature_id': str(fid),
            'abundance': str(abd),
            'sample_metadata_fields': sample_fields,
            'feature_metadata_fields': feature_fields,
            'sample_metadata_values': sample_values,
            'feature_metadata_values': feature_values,
            'annotations': self._format_annotations(annt),
        }
        return self._last_info

    @staticmethod
    def _annotation_color(annotation_type):
        if annotation_type == 'diffexp':
            return '#1f4ed8'
        if annotation_type == 'contamination':
            return '#b91c1c'
        if annotation_type in ('common', 'highfreq'):
            return '#047857'
        return '#111827'

    def _format_annotations(self, annt):
        results = []
        for item in annt:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                details = item[0] if isinstance(item[0], dict) else {}
                text = str(item[1])
                atype = details.get('annotationtype', '')
                url = None
                if '_db_interface' in details:
                    try:
                        url = details['_db_interface'].get_annotation_website(details)
                    except Exception:
                        url = None
                results.append({
                    'text': text,
                    'annotation_type': atype,
                    'color': self._annotation_color(atype),
                    'url': url,
                })
            else:
                results.append({
                    'text': str(item),
                    'annotation_type': '',
                    'color': self._annotation_color(''),
                    'url': None,
                })
        return results

    def _render_side_panel_html(self):
        fig_url = html.escape(self._figure_url or '')
        fig_ws_url = html.escape((self._figure_url or '').replace('http://', 'ws://').rstrip('/') + '/ws')
        return self._get_side_panel_template().safe_substitute(fig_url=fig_url, fig_ws_url=fig_ws_url)

    @classmethod
    def _get_side_panel_template(cls):
        if cls._side_panel_template is None:
            template_path = Path(__file__).with_name('plotgui_webagg_template.html')
            with open(template_path, encoding='utf-8') as fp:
                cls._side_panel_template = Template(fp.read())
        return cls._side_panel_template

    def show_info(self):
        # Update cache for the companion side panel, and keep CLI fallback logs.
        info = self._collect_side_info()
        logger.debug('selected sample=%s feature=%s abundance=%s',
                     info['sample_id'], info['feature_id'], info['abundance'])
