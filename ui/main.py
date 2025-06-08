import os
import time
import gradio as gr
import threading
import queue
from typing import Dict, Any, List
import numpy as np
import traceback
from funsearch import function, archipelago, cluster, presenter, slack
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


def run_funsearch_process(formula: str, params: str, data: str, insights: str,
                          max_nparams: int, max_mutations: int, auto_cleanup: bool, request: gr.Request):
    """Gradio から呼び出され、FunSearch を実行し、結果を yield する。"""
    session_hash = request.session_hash
    if not session_hash:
        yield "Error: No session hash found.\n", UPDATE_HEADER
        return

    sessions[session_hash] = {
        'cancelled': False,
        'evolver': None,
        'worker_thread': None,
        'auto_cleanup': auto_cleanup
    }

    q = queue.Queue()
    full_log = ""
    update_list: List[str] = []

    # Slack通知用の設定
    notifier = None
    try:
        notifier = slack.SlackNotifier()
    except ValueError:
        # Slack設定が無い場合は通知無しで続行
        pass

    start_time = time.time()

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
        target=presenter.funsearch_worker,
        args=(q, formula, params, insights, inputs_np,
              outputs_np, max_nparams, max_mutations, sessions[session_hash], GEMINI_CLIENT_FOR_CONVERTER, notifier, start_time),
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

    # auto_cleanupが無効の場合は何もしない
    if not session_data.get('auto_cleanup', True):
        return

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
            auto_cleanup_checkbox = gr.Checkbox(
                label="ページ離脱時に自動停止", value=True,
                info="チェックを外すと、ページを離脱してもバックグラウンドで実行が継続されます。")

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
                insights_input, max_nparams_input, max_mutations_input, auto_cleanup_checkbox],
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
    # IAP認証を使用するため、Gradio Basic認証は無効化
    # gradio_user = os.environ.get("GRADIO_USER", "qunasys")
    # gradio_pass = os.environ.get("GRADIO_PASSWORD")
    # if not gradio_pass:
    #     gradio_pass = ''.join(secrets.choice(
    #         string.ascii_letters + string.digits) for _ in range(16))
    #     print(
    #         f"No GRADIO_PASSWORD env var. Using generated password for user '{gradio_user}': {gradio_pass}")
    # else:
    #     print(
    #         f"Using password from GRADIO_PASSWORD env var for user '{gradio_user}'.")

    # auth_creds = (gradio_user, gradio_pass) if gradio_pass else None
    auth_creds = None  # IAP認証を使用するため無効化
    demo.launch(auth=auth_creds, share=True, server_name="0.0.0.0")
