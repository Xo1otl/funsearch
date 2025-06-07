import os
import secrets
import string
import gradio as gr
import time
import threading
import queue
from typing import Dict, Any, List
import numpy as np
import traceback
from funsearch import llmsr, datadriven, function, archipelago, cluster
from google import genai

AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent

try:
    api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
except KeyError:
    from infra.ai import llm
    api_key = llm.GOOGLE_CLOUD_API_KEY

GEMINI_CLIENT_FOR_CONVERTER = genai.Client(api_key=api_key)
UPDATE_HEADER = "## Best Functions Found:\n\n"

sessions: Dict[str, Dict[str, Any]] = {}


class CancellableInputConverter:
    def __init__(self, client, session_hash: str):
        self.client = client
        self.session_hash = session_hash
        self.original_converter = datadriven.InputConverter(client)

    def convert(self, formula: str, specs: str, insights: str):
        if self._is_cancelled():
            raise InterruptedError("Conversion cancelled")
        return self.original_converter.convert(formula, specs, insights)

    def _is_cancelled(self) -> bool:
        return sessions.get(self.session_hash, {}).get('cancelled', False)


class DetailedProfiler:
    def __init__(self, output_queue: queue.Queue, max_mutations: int, session_hash: str):
        self.q = output_queue
        self.evaluation_count = 0
        self.mutation_count = 0
        self.max_mutations = max_mutations
        self.session_hash = session_hash
        self.start_times_eval: Dict[int, float] = {}
        self.start_times_mutate: Dict[int, float] = {}

    def _check_stop_conditions(self) -> bool:
        # キャンセルチェックを追加
        if sessions.get(self.session_hash, {}).get('cancelled', False):
            self._stop_evolver()
            self.q.put(('log', "--- Process cancelled by user. ---\n"))
            self.q.put(('stop', 'Cancelled by user'))
            return True

        if self.max_mutations > 0 and self.mutation_count >= self.max_mutations:
            self._stop_evolver()
            self.q.put(
                ('log', f"--- Max mutations ({self.max_mutations}) reached. Stopping evolver. ---\n"))
            self.q.put(('stop', 'Max mutations reached'))
            return True

        return False

    def _stop_evolver(self):
        """Evolverを停止"""
        session_data = sessions.get(self.session_hash, {})
        evolver = session_data.get('evolver')
        if evolver is not None:
            evolver.stop()
            session_data['evolver'] = None

    def _format_function(self, fn: Any) -> str:
        return str(fn.skeleton())

    def _get_score(self, fn_or_payload: Any) -> str:
        try:
            score = None
            if hasattr(fn_or_payload, 'score') and callable(fn_or_payload.score):
                score = fn_or_payload.score()
            elif isinstance(fn_or_payload, tuple) and len(fn_or_payload) > 1:
                score = fn_or_payload[1]
            return f"{score}" if score is not None else "N/A"
        except Exception:
            return "?.???"

    def profile(self, event: AllEvent):
        if self._check_stop_conditions():
            return

        message = ""
        body = ""
        current_time = time.perf_counter()
        thread_id = threading.get_ident()

        if event.type == "on_evaluate":
            self.start_times_eval[thread_id] = current_time
            message = "Starting evaluation..."
        elif event.type == "on_evaluated":
            elapsed_time = -1.0
            start_time = self.start_times_eval.pop(thread_id, None)
            if start_time is not None:
                elapsed_time = current_time - start_time
            self.evaluation_count += 1
            message = f"Evaluation finished in {elapsed_time:.4f}s. Score: {self._get_score(event.payload)}"
        elif event.type == "on_best_island_improved":
            count = self.evaluation_count
            message = "✨ Best island function improved!"
            best_fn = event.payload.best_fn()
            score = self._get_score(best_fn)
            code = self._format_function(best_fn)
            title = " Evaluated Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}"
            update_message = f"**Score: {score}** (Eval: {count})\n\n```python\n{code}\n```\n\n---\n\n"
            self.q.put(('update', update_message))
        elif event.type == "on_best_fn_improved":
            count = self.evaluation_count
            message = "🏝️ Best function improved (within island)!"
            best_fn = event.payload
            score = self._get_score(best_fn)
            code = self._format_function(best_fn)
            title = " Island Best Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}"
        elif event.type == "on_islands_removed":
            message = f"Removed islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_islands_revived":
            message = f"Revived islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_fn_added":
            message = f"New function added. Score: {self._get_score(event.payload)}"
        elif event.type == "on_fn_selected":
            lengths = [len(self._format_function(fn))
                       for fn in event.payload[0]]
            message = f"Selected fn. Lengths: {lengths}. Score: {self._get_score(event.payload[1])}"
        elif event.type == "on_mutate":
            self.start_times_mutate[thread_id] = current_time
            scores = [self._get_score(fn) for fn in event.payload]
            message = f"Starting mutation. Scores: {scores}"
        elif event.type == "on_mutated":
            elapsed_time = -1.0
            start_time = self.start_times_mutate.pop(thread_id, None)
            if start_time is not None:
                elapsed_time = current_time - start_time
            self.mutation_count += 1
            count = self.mutation_count
            scores = [self._get_score(fn) for fn in event.payload[0]]
            message = f"Mutation finished in {elapsed_time:.4f}s. Scores: {scores}"

        if message:
            log_message = f"| {event.type:<20} | {message}{body}\n"
            self.q.put(('log', log_message))


def funsearch_worker(q: queue.Queue, formula: str, specs: str, insights: str,
                     inputs: np.ndarray, outputs: np.ndarray, max_nparams: int,
                     max_mutations: int, session_hash: str):
    evolver = None
    try:
        if sessions.get(session_hash, {}).get('cancelled', False):
            q.put(('log', "--- Process cancelled before starting. ---\n"))
            return

        q.put(('log', "--- Starting FunSearch Worker ---\n"))

        converter = CancellableInputConverter(
            GEMINI_CLIENT_FOR_CONVERTER, session_hash)
        q.put(('log', "2. Calling LLM to convert input...\n"))

        info = converter.convert(formula, specs, insights)

        if not info or not info.get("equation_src"):
            q.put(('log', "| Error | InputConverter failed or returned empty source.\n"))
            return

        if sessions.get(session_hash, {}).get('cancelled', False):
            q.put(('log', "--- Process cancelled after conversion. ---\n"))
            return

        q.put(
            ('log', f"--- Generated Code ---\n{info['equation_src']}\n---\n"))

        profiler = DetailedProfiler(q, max_mutations, session_hash)

        datasets = [datadriven.Dataset(max_nparams, inputs, outputs)]
        q.put(('log', "3. Starting FunSearch evolver...\n"))
        q.put(('log', "=" * 70 + "\n"))

        evolver_config = llmsr.EvolverConfigForMCP(
            equation_src=info["equation_src"], docstring=info["docstring"],
            evaluation_inputs=datasets, evaluator=datadriven.dataset_evaluator,
            prompt_comment=info["prompt_comment"], profiler_fn=profiler.profile,
            max_nparams=max_nparams
        )
        evolver = llmsr.spawn_evolver_for_mcp(evolver_config)

        if session_hash in sessions:
            sessions[session_hash]['evolver'] = evolver

        if sessions.get(session_hash, {}).get('cancelled', False):
            q.put(('log', "--- Process cancelled before evolver start. ---\n"))
            return

        q.put(('log', "--- Evolver starting evolution process. ---\n"))
        evolver.start()
        q.put(('log', "\n" + "=" * 70 + "\n"))
        q.put(('log', "4. FunSearch evolver finished.\n"))

    except InterruptedError as e:
        q.put(('log', f"--- Process interrupted: {e} ---\n"))
    except Exception as e:
        q.put(('log', f"| Error (Worker) | {e}\n{traceback.format_exc()}\n"))
    finally:
        if session_hash in sessions:
            session_data = sessions[session_hash]
            if 'evolver' in session_data:
                session_data['evolver'] = None
        q.put(('end', None))


def run_funsearch_process(formula: str, params: str, data: str, insights: str,
                          max_nparams: int, max_mutations: int, request: gr.Request):
    """Gradio から呼び出され、FunSearch を実行し、結果を yield する。"""
    session_hash = request.session_hash
    if not session_hash:
        yield "Error: No session hash found.\n", UPDATE_HEADER
        return

    sessions[session_hash] = {
        'cancelled': False,
        'evolver': None,
        'worker_thread': None
    }

    q = queue.Queue()
    full_log = ""
    update_list: List[str] = []

    if not all([formula, params, data]):
        yield "Error: Please provide Formula, Parameters, and Data.\n", UPDATE_HEADER
        return

    try:
        lines = [list(map(float, line.split(',')))
                 for line in data.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("No data points found or data is empty.")
        inputs_np = np.array([l[:-1] for l in lines])
        outputs_np = np.array([l[-1] for l in lines])
        full_log += f"1. Parsed {len(lines)} data points.\n"
    except Exception as e:
        yield f"Error parsing data: {e}\n{traceback.format_exc()}\n", UPDATE_HEADER
        return

    yield full_log, UPDATE_HEADER + "".join(update_list)

    worker_thread = threading.Thread(
        target=funsearch_worker,
        args=(q, formula, params, insights, inputs_np,
              outputs_np, max_nparams, max_mutations, session_hash),
        daemon=True
    )

    sessions[session_hash]['worker_thread'] = worker_thread
    worker_thread.start()

    while True:
        try:
            msg_type, content = q.get(timeout=1.0)

            if msg_type == 'end':
                full_log += "--- FunSearch process ended. ---\n"
                break
            elif msg_type == 'log':
                full_log += str(content)
            elif msg_type == 'update':
                update_list.insert(0, str(content))
            elif msg_type == 'stop':
                full_log += f"--- Evolution stopped: {content} ---\n"

            yield full_log, UPDATE_HEADER + "".join(update_list)

        except queue.Empty:
            if not worker_thread.is_alive():
                full_log += "--- Worker thread ended. ---\n"
                break

    full_log += "--- FunSearch process completed. ---\n"

    if session_hash in sessions:
        del sessions[session_hash]

    yield full_log, UPDATE_HEADER + "".join(update_list)


def stop_funsearch_process(request: gr.Request):
    """FunSearchプロセスを停止"""
    session_hash = request.session_hash
    if not session_hash:
        gr.Error("Error: No session hash found.")
        return

    if session_hash not in sessions:
        gr.Warning("停止するプロセスが見つかりませんでした。")
        return

    session_data = sessions[session_hash]

    session_data['cancelled'] = True

    evolver = session_data.get('evolver')
    if evolver is not None:
        evolver.stop()
        session_data['evolver'] = None
        gr.Info("Evolverを停止しました。")
    else:
        gr.Info("プロセスをキャンセルしました。")


def cleanup_session(request: gr.Request):
    session_hash = request.session_hash
    if not session_hash or session_hash not in sessions:
        return

    session_data = sessions[session_hash]

    session_data['cancelled'] = True

    evolver = session_data.get('evolver')
    if evolver is not None:
        evolver.stop()
        session_data['evolver'] = None


default_formula = r'''このモデルは、粒子で充填されたゴム複合材料の引張弾性率を予測することを目的としています。
特に、充填材の体積分率（phi）と複合材料の引張弾性率（E_composite）の関係をモデル化します。
基礎となる物理モデルは、複合材料の弾性率を定義するReussモデルです。
Reussモデルは以下の式で表されます。

E_composite = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)

ここで、E_mはマトリックスの引張弾性率であり、4.84で固定されています。
E_fは充填材の引張弾性率であり、117.64で固定されています。'''
default_params = r'\phi: フィラー体積分率'
default_data = '0,4.84\n0.09,5.56\n0.17,6.13\n0.33,10.13\n0.44,14.96'
default_insights = '''あなたのタスクは与えられた変数（phi, E_m, E_f）を用いて、複合材料の引張弾性率 E_composite を予測する関数 E_composite = f(phi, params, E_m, E_f) を進化させることです。
進化の出発点は提供されたReussモデルです。
最大で MAX_NPARAMS 個の最適化可能なパラメータ（params 配列から）を導入して、自由に改良を重ねて、実験データとの適合性を向上させることを目指してください。
最終的な目標は、基本的なReussモデルに対して、物理的に意味のある改善を見つけ出すことです。'''
default_nparams = 1
default_max_mutations = 50

with gr.Blocks(theme=gr.themes.Soft()) as demo:  # type: ignore
    gr.Markdown("# FunSearch Gradio Interface (With Enhanced Stop Button)")

    with gr.Row():
        with gr.Column(scale=1):
            formula_input = gr.Textbox(
                lines=5, label="理論式", value=default_formula)
            params_input = gr.Textbox(
                lines=2, label="パラメータ説明", value=default_params)
            data_input = gr.Textbox(
                lines=5, label="データ (CSV)", value=default_data)
            insights_input = gr.Textbox(
                lines=3, label="着眼点", value=default_insights)
            max_nparams_input = gr.Number(
                label="最大パラメータ数", value=default_nparams, precision=0, step=1,
                info="進化の過程で追加できる最大のパラメータ数を指定します。")
            max_mutations_input = gr.Number(
                label="変異回数", value=default_max_mutations, precision=0, step=1,
                info="回数に達するまで停止しないので必ず適切な値を設定してください。")

            with gr.Row():
                run_button = gr.Button("実行", variant="primary")
                stop_button = gr.Button("停止", variant="stop")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="実行ログ", lines=25, autoscroll=True, show_copy_button=True)
            update_output = gr.Markdown(label="更新ログ (Best Functions)")

    run_event = run_button.click(
        fn=run_funsearch_process,
        inputs=[formula_input, params_input, data_input,
                insights_input, max_nparams_input, max_mutations_input],
        outputs=[log_output, update_output],
        show_progress="full",
        concurrency_limit=2
    )

    stop_button.click(
        fn=stop_funsearch_process,
        inputs=None,
    )

    demo.unload(
        fn=cleanup_session,
    )

if __name__ == "__main__":
    print("Launching Gradio UI...")
    gradio_user = os.environ.get("GRADIO_USER", "qunasys")
    gradio_pass = os.environ.get("GRADIO_PASSWORD")
    if not gradio_pass:
        gradio_pass = ''.join(secrets.choice(
            string.ascii_letters + string.digits) for _ in range(16))
        print(
            f"No GRADIO_PASSWORD env var. Using generated password for user '{gradio_user}': {gradio_pass}")
    else:
        print(
            f"Using password from GRADIO_PASSWORD env var for user '{gradio_user}'.")

    auth_creds = (gradio_user, gradio_pass) if gradio_pass else None
    demo.launch(auth=auth_creds, share=True, server_name="0.0.0.0")
