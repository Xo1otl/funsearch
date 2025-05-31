import os
import secrets
import string
import gradio as gr
import time
import threading
import queue
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import traceback
from funsearch import llmsr, datadriven, function, archipelago, cluster
from google import genai
from infra.ai import llm

# --- 定数 ---
AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent
gemini_client_for_converter = genai.Client(api_key=llm.GOOGLE_CLOUD_API_KEY)
UPDATE_HEADER = "## Best Functions Found:\n\n"

# --- グローバル変数 ---
current_evolver: Optional[archipelago.Evolver] = None
evolver_lock = threading.Lock()
global_queue: Optional[queue.Queue] = None


class DetailedProfiler:
    def __init__(self, output_queue: queue.Queue, max_mutations):
        self.q = output_queue
        self._evaluation_count = 0
        self._lock = threading.Lock()
        self._start_times_eval: Dict[int, float] = {}
        self._start_times_mutate: Dict[int, float] = {}
        self.max_mutations = max_mutations  # <<< 最大変異回数を保持
        self._mutation_count = 0

    def _format_function(self, fn: Any) -> str:
        try:
            return str(fn.skeleton())
        except Exception:
            return "[Func Display Error]"

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
        message = ""
        body = ""
        current_time = time.perf_counter()
        thread_id = threading.get_ident()

        try:
            if event.type == "on_evaluate":
                with self._lock:
                    self._start_times_eval[thread_id] = current_time
                message = "Starting evaluation..."
            elif event.type == "on_evaluated":
                elapsed_time = -1.0
                with self._lock:
                    start_time = self._start_times_eval.pop(thread_id, None)
                    if start_time is not None:
                        elapsed_time = current_time - start_time
                    self._evaluation_count += 1
                message = f"Evaluation finished in {elapsed_time:.4f}s. Score: {self._get_score(event.payload)}"
            elif event.type == "on_best_island_improved":
                with self._lock:
                    count = self._evaluation_count
                message = "✨ Best island function improved!"
                best_fn = event.payload.best_fn()
                score = self._get_score(best_fn)
                code = self._format_function(best_fn)
                title = " Evaluated Function "
                padding = (60 - len(title)) // 2
                formatted_title = "=" * padding + title + \
                    "=" * (60 - len(title) - padding)
                body = (
                    f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}")
                update_message = f"**Score: {score}** (Eval: {count})\n\n```python\n{code}\n```\n\n---\n\n"
                self.q.put(('update', update_message))  # Send update message
            elif event.type == "on_best_fn_improved":
                with self._lock:  # 評価カウントを安全に取得
                    count = self._evaluation_count
                # メッセージを更新
                message = "🏝️ Best function improved (within island)!"
                best_fn = event.payload
                score = self._get_score(best_fn)
                code = self._format_function(best_fn)
                title = " Island Best Function "  # タイトルを少し変更
                padding = (60 - len(title)) // 2
                formatted_title = "=" * padding + title + \
                    "=" * (60 - len(title) - padding)
                # on_best_island_improved と同様のフォーマットで body を生成
                body = (
                    f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}")
                # ここでは ('update', ...) は送信しません (実行ログのみ)
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
                scores = [self._get_score(fn) for fn in event.payload]
                message = f"Starting mutation. Scores: {scores}"
                with self._lock:
                    self._start_times_mutate[thread_id] = current_time
            elif event.type == "on_mutated":
                should_stop = False
                elapsed_time = -1.0
                with self._lock:
                    start_time = self._start_times_mutate.pop(thread_id, None)
                    if start_time is not None:
                        elapsed_time = current_time - start_time

                    self._mutation_count += 1  # カウントアップ
                    count = self._mutation_count

                    # 最大回数に達したかチェック
                    if self.max_mutations and count >= self.max_mutations:
                        should_stop = True

                scores = [self._get_score(fn) for fn in event.payload[0]]
                message = f"Mutation finished in {elapsed_time:.4f}s. Scores: {scores}"

                # 最大回数に達していれば、停止リクエストをキューに送る
                if should_stop:
                    stop_msg = f"\n--- Max mutations ({self.max_mutations}) reached. Requesting stop. ---"
                    message += stop_msg
                    self.q.put(('stop_request', 'Max mutations reached'))

            if message:  # Only put if there is a message
                log_message = f"| {event.type:<20} | {message}{body}\n"
                self.q.put(('log', log_message))

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"| Profiler Error          | {e} | {event.type}\n{tb_str}\n"
            self.q.put(('log', error_message))


def stop_funsearch_process():
    """FunSearch プロセスを停止する。"""
    global current_evolver, global_queue
    with evolver_lock:
        if current_evolver:
            try:
                if global_queue:
                    global_queue.put(
                        ('log', "--- Sending stop signal... ---\n"))
                current_evolver.stop()
                if global_queue:
                    global_queue.put(('log', "--- Stop signal sent. ---\n"))
                current_evolver = None
            except Exception as e:
                if global_queue:
                    global_queue.put(('log', f"| Error (Stop) | {e}\n"))
        elif global_queue:
            global_queue.put(('log', "| Info | No process running.\n"))


def funsearch_worker(q: queue.Queue, formula, specs, insights, inputs, outputs, max_nparams, max_mutations):
    """FunSearch を実行するワーカースレッド。"""
    global current_evolver
    try:
        q.put(('log', "--- Starting FunSearch Worker ---\n"))
        converter = datadriven.InputConverter(gemini_client_for_converter)
        q.put(('log', "2. Calling LLM...\n"))
        info = converter.convert(formula, specs, insights)
        if not info:
            q.put(('log', "| Error | InputConverter failed.\n"))
            return

        q.put(
            ('log', f"--- Generated Code ---\n{info['equation_src']}\n---\n"))
        profiler = DetailedProfiler(q, max_mutations)
        datasets = [datadriven.Dataset(max_nparams, inputs, outputs)]
        q.put(('log', "3. Starting FunSearch...\n"))
        q.put(('log', "=" * 70 + "\n"))

        evolver = llmsr.spawn_evolver_for_mcp(
            llmsr.EvolverConfigForMCP(
                equation_src=info["equation_src"], docstring=info["docstring"],
                evaluation_inputs=datasets, evaluator=datadriven.dataset_evaluator,
                prompt_comment=info["prompt_comment"], profiler_fn=profiler.profile,
                max_nparams=max_nparams))

        with evolver_lock:
            current_evolver = evolver
        evolver.start()
        q.put(('log', "=" * 70 + "\n"))
        q.put(('log', "4. FunSearch finished or stopped!\n"))

    except Exception as e:
        q.put(('log', f"| Error (Worker) | {e}\n{traceback.format_exc()}\n"))
    finally:
        with evolver_lock:
            current_evolver = None
        q.put(('end', None))


def run_funsearch_process(formula, params, data, insights, max_nparams, max_mutations):
    """Gradio から呼び出され、FunSearch を実行し、結果を yield する。"""
    global global_queue
    q = queue.Queue()
    global_queue = q

    full_log = ""
    update_list: List[str] = []

    if not all([formula, params, data]):
        yield "Error: Please provide all inputs.\n", ""
        return

    try:
        lines = [list(map(float, line.split(',')))
                 for line in data.strip().split('\n')]
        inputs_np = np.array([l[:-1] for l in lines])
        outputs_np = np.array([l[-1] for l in lines])
        full_log += f"1. Parsed {len(lines)} data points.\n"
    except Exception as e:
        yield f"Error parsing data: {e}\n", ""
        return

    yield full_log, UPDATE_HEADER

    threading.Thread(
        target=funsearch_worker,
        args=(q, formula, params, insights,
              inputs_np, outputs_np, max_nparams, max_mutations),
        daemon=True,
    ).start()

    while True:
        try:
            msg_type, content = q.get(timeout=1.0)

            if msg_type == 'end':
                break
            elif msg_type == 'log':
                full_log += content
            elif msg_type == 'update':
                update_list.insert(0, content)
            elif msg_type == 'stop_request':
                full_log += f"--- Stop requested via queue: {content} ---\n"
                stop_funsearch_process()
                break

            yield full_log, UPDATE_HEADER + "".join(update_list)

        except queue.Empty:
            # ワーカーが'end'を送らずに終了した場合のフォールバック
            if not current_evolver and not any(
                    t for t in threading.enumerate() if t.daemon and t._target == funsearch_worker):  # type: ignore
                full_log += "--- Worker thread seems to have ended unexpectedly. ---\n"
                break

    global_queue = None
    full_log += "--- Process Ended ---\n"
    yield full_log, UPDATE_HEADER + "".join(update_list)


# --- デフォルト入力値 (変更なし) ---
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
default_nparams = 1  # 最大パラメータ数

# --- Gradio UI 構築 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:  # type: ignore
    gr.Markdown("# FunSearch Gradio Interface (Detailed Log)")

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
                info="進化の仮定で追加できる最大のパラメータ数を指定します。")
            max_mutations = gr.Number(
                label="最大変異数", value=default_nparams, precision=0, step=1,
                info="最大の変異回数、大きすぎるとAPIの料金が増えます")
            run_button = gr.Button("実行", variant="primary")
            stop_button = gr.Button("停止", variant="stop")
        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="実行ログ", lines=20, autoscroll=True, show_copy_button=True)  # <<< lines を増やした
            update_output = gr.Markdown(label="更新ログ (Best Functions)")

    run_button.click(
        fn=run_funsearch_process,
        inputs=[formula_input, params_input, data_input,
                insights_input, max_nparams_input, max_mutations],
        outputs=[log_output, update_output],
    )
    stop_button.click(fn=stop_funsearch_process, inputs=None, outputs=None)

# --- Gradio アプリケーションの起動 ---
if __name__ == "__main__":
    print("Launching Gradio UI...")
    password = os.environ.get("GRADIO_PASSWORD") or ''.join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
    print(f"Using password: {password}")
    demo.launch(auth=("qunasys", password), share=True)
    # demo.launch(share=True)
