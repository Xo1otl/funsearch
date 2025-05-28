import gradio as gr
import time
import threading
import queue
from typing import Dict, Any, Optional
import numpy as np
import traceback
from funsearch import llmsr
from funsearch import datadriven
from funsearch import function
from funsearch import archipelago
from funsearch import cluster
# Google AI 関連 (ユーザー指定の形式)
from google import genai
from infra.ai import llm  # ユーザー指定のモジュール
AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent

gemini_client_for_converter: Optional[genai.Client] = genai.Client(
    api_key=llm.GOOGLE_CLOUD_API_KEY)


class MyGradioProfiler:
    """
    FunSearch のイベントを受け取り、整形したログメッセージを Queue に入れるクラス。
    ユーザー提供の Profiler コードをベースにしています。
    """

    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self._evaluation_count = 0
        self._lock = threading.Lock()
        self._start_times_eval: Dict[int, float] = {}
        self._start_times_mutate: Dict[int, float] = {}

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
            return f"{score:.4f}" if score is not None else "N/A"
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
                    current_eval_count = self._evaluation_count
                message = "✨ Best island function improved!"
                best_fn = event.payload.best_fn()
                title = " Evaluated Function "
                padding = (60 - len(title)) // 2
                formatted_title = "=" * padding + title + \
                    "=" * (60 - len(title) - padding)
                body = (f"""
{formatted_title}
{self._format_function(best_fn)}
{'-' * 60}
Score      : {self._get_score(best_fn)}
Evaluations: {current_eval_count}
{'=' * 60}
""")
            elif event.type == "on_best_fn_improved":
                message = "Best function improved (within island)!"
                title = " Evaluated Function "
                padding = (60 - len(title)) // 2
                formatted_title = "=" * padding + title + \
                    "=" * (60 - len(title) - padding)
                body = (f"""
{formatted_title}
{self._format_function(event.payload)}
{'-' * 60}
Score: {self._get_score(event.payload)}
{'=' * 60}
""")
            elif event.type == "on_islands_removed":
                message = f"Removed islands: {[hex(id(island)) for island in event.payload]}"
            elif event.type == "on_islands_revived":
                message = f"Revived islands: {[hex(id(island)) for island in event.payload]}"
            elif event.type == "on_fn_added":
                message = f"New function added. Score: {self._get_score(event.payload)}"
            elif event.type == "on_fn_selected":
                code_lengths_str = ", ".join(
                    map(str, [len(self._format_function(fn)) for fn in event.payload[0]]))
                message = f"Selected function. Lengths: [{code_lengths_str}]. Score: {self._get_score(event.payload[1])}"
            elif event.type == "on_mutate":
                scores_str = ", ".join([self._get_score(fn)
                                       for fn in event.payload])
                message = f"Starting mutation. Scores: [{scores_str}]"
                with self._lock:
                    self._start_times_mutate[thread_id] = current_time
            elif event.type == "on_mutated":
                elapsed_time = -1.0
                with self._lock:
                    start_time = self._start_times_mutate.pop(thread_id, None)
                if start_time is not None:
                    elapsed_time = current_time - start_time
                scores_str = ", ".join([self._get_score(fn)
                                       for fn in event.payload[0]])
                message = f"Mutation finished in {elapsed_time:.4f}s. Scores: [{scores_str}]"

            body_section = f"\n{body}" if body else ""
            log_message = f"| {event.type:<20} | {message}{body_section}\n"
            self.output_queue.put(log_message)

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"| Profiler Error          | Error: {e} | Event: {getattr(event, 'type', 'unknown')}\n{tb_str}\n"
            self.output_queue.put(error_message)


# --- FunSearch 実行ワーカー ---
current_evolver = None
evolver_lock = threading.Lock()
log_queue_global: Optional[queue.Queue] = None


def stop_funsearch_process():
    """Stops the currently running FunSearch process."""
    global current_evolver
    global log_queue_global

    with evolver_lock:
        if current_evolver:
            try:
                if log_queue_global:
                    log_queue_global.put("--- Sending stop signal... ---\n")
                current_evolver.stop()
                if log_queue_global:
                    log_queue_global.put(
                        "--- Stop signal sent successfully. ---\n")
            except Exception as e:
                if log_queue_global:
                    log_queue_global.put(
                        f"| Error (Stop)          | Failed to send stop signal: {e}\n")
        else:
            # If the queue exists, send a message. Otherwise, print.
            if log_queue_global:
                log_queue_global.put(
                    "| Info                  | No FunSearch process is currently running.\n")
            else:
                print("Stop clicked, but no process or queue found.")


def funsearch_worker(
    q: queue.Queue,
    formula_text: str,
    variables_specs: str,
    insights_text: str,
    inputs: np.ndarray,
    outputs: np.ndarray,
):
    global current_evolver
    global log_queue_global
    """FunSearch を実行し、ログを Queue に送信する関数。"""
    try:
        q.put("--- Starting FunSearch Worker ---\n")

        if not gemini_client_for_converter:
            q.put("| Error                   | Google AI Client not available.\n")
            q.put(None)
            return

        # 2. InputConverter
        converter = datadriven.InputConverter(gemini_client_for_converter)
        q.put("2. Calling LLM via InputConverter...\n")
        input_info = converter.convert(
            formula_text=formula_text,
            variables_specs=variables_specs,
            insights_text=insights_text,
        )
        if input_info is None:
            q.put(
                "| Error                   | Failed to convert inputs via InputConverter.\n")
            q.put(None)
            return

        q.put("   - Code generated successfully.\n")
        max_nparams = 1
        equation_src = input_info["equation_src"]
        docstring = input_info["docstring"]
        prompt_comment = input_info["prompt_comment"]
        q.put(
            f"--- Generated Code ---\n{equation_src}\n----------------------\n")

        # 3. FunSearch 実行
        profiler = MyGradioProfiler(q)
        datasets = [datadriven.Dataset(
            max_nparams=max_nparams, inputs=inputs, outputs=outputs)]
        q.put("3. Starting FunSearch process (this may take time)...\n")
        q.put("=" * 70 + "\n")

        evolver = llmsr.spawn_evolver_for_mcp(
            llmsr.EvolverConfigForMCP(
                equation_src=equation_src,
                docstring=docstring,
                evaluation_inputs=datasets,
                evaluator=datadriven.dataset_evaluator,
                prompt_comment=prompt_comment,
                profiler_fn=profiler.profile,  # カスタム Profiler を渡す
                max_nparams=max_nparams,
            )
        )

        with evolver_lock:  # <<< 追加
            current_evolver = evolver  # Set global evolver # <<< 追加

        evolver.start()

        q.put("=" * 70 + "\n")
        q.put("4. FunSearch finished!\n")

    except Exception as e:
        tb_str = traceback.format_exc()
        q.put(
            f"| Error (Worker)          | An error occurred: {e}\n{tb_str}\n")
    finally:
        q.put(None)


def run_funsearch_process(formula, params, data, insights):
    log_queue = queue.Queue()
    full_log = ""

    if not all([formula, params, data]):
        yield "Error: Please provide Formula, Parameters, and Data.\n"
        return

    full_log += "1. Parsing input data...\n"
    yield full_log
    time.sleep(0.1)

    try:
        inputs_list, outputs_list = [], []
        for line in data.strip().split('\n'):
            parts = [float(p.strip()) for p in line.split(',')]
            inputs_list.append(parts[:-1])
            outputs_list.append(parts[-1])
        inputs_np, outputs_np = np.array(inputs_list), np.array(outputs_list)
        full_log += f"   - Found {len(inputs_list)} data points.\n"
        yield full_log
    except Exception as e:
        full_log += f"   - Error parsing data: {e}.\n"
        yield full_log
        return

    runner_thread = threading.Thread(
        target=funsearch_worker,
        args=(log_queue, formula, params, insights, inputs_np, outputs_np),
        daemon=True,
    )
    runner_thread.start()

    while True:
        try:
            log_entry = log_queue.get(timeout=0.1)
            if log_entry is None:
                break
            full_log += log_entry
            yield full_log
        except queue.Empty:
            if not runner_thread.is_alive():
                break

    runner_thread.join(timeout=1)
    full_log += "--- Process Ended ---\n"
    yield full_log


# --- デフォルト入力値 (元のコードから) ---
default_formula = r'''理論式(latexでも可)とその説明を自由に記述してください。'''
default_formula = r'''
このモデルは、粒子で充填されたゴム複合材料の引張弾性率を予測することを目的としています。
特に、充填材の体積分率（phi）と複合材料の引張弾性率（E_composite）の関係をモデル化します。
基礎となる物理モデルは、複合材料の弾性率を定義するReussモデルです。
Reussモデルは以下の式で表されます。

E_composite = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)

ここで、E_mはマトリックスの引張弾性率であり、4.84で固定されています。
E_fは充填材の引張弾性率であり、117.64で固定されています。
'''
default_params = '''\
変数名1: 説明1
変数名2: 説明2
出力変数名: 関数が出力する値の説明を書いてください、これは必ず最後の行に書いてください\
'''
default_params = r'''
\phi: フィラー体積分率 (実験で扱う入力)
'''
default_data = '''\
0,4.84
0.09,5.56
0.17,6.13
0.33,10.13
0.44,14.96
'''
default_insights = r'''その他の着眼点やヒントを自由に記述してください。'''
default_insights = r'''
与えられた変数（phi, E_m, E_f）を用いて、複合材料の引張弾性率 E_composite を予測する関数 E_composite = f(phi, params, E_m, E_f) を進化させることです。
進化の出発点は提供されたReussモデルとします。
最大で MAX_NPARAMS 個の最適化可能なパラメータ（params 配列から）を導入して、Reussモデルを修正または拡張し、実験データとの適合性を向上させることを目指してください。
例えば、モデルをスケーリングしたり、新しい項を追加したり、モデルの構成要素を変更したりといったアプローチが考えられます。
最終的な目標は、基本的なReussモデルに対して、物理的に意味のある改善を見つけ出すことです。
'''

# --- Gradio UI 構築 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:  # type: ignore
    gr.Markdown("# FunSearch Gradio Interface (Actual Run)")
    status = "Ready."
    gr.Markdown(f"**Status:** {status}")

    with gr.Row():
        with gr.Column(scale=1):
            formula_input = gr.Textbox(
                lines=8, label="理論式", value=default_formula)
            params_input = gr.Textbox(
                lines=6, label="パラメータ説明", value=default_params)
            data_input = gr.Textbox(
                lines=8, label="データ (CSV)", value=default_data)
            insights_input = gr.Textbox(
                lines=5, label="着眼点", value=default_insights)
            run_button = gr.Button("FunSearch 実行", variant="primary",
                                   interactive=True)
            stop_button = gr.Button("FunSearch 停止", variant="stop",
                                    interactive=True)

        with gr.Column(scale=2):
            log_output = gr.Textbox(label="実行ログ", lines=30, interactive=False,
                                    autoscroll=True, show_copy_button=True)

    run_button.click(
        fn=run_funsearch_process,
        inputs=[formula_input, params_input, data_input, insights_input],
        outputs=[log_output],
    )

    stop_button.click(  # <<< 追加 (ここから3行)
        fn=stop_funsearch_process,
        inputs=None,
        outputs=None,
    )
# --- Gradio アプリケーションの起動 ---
if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch(share=True)
