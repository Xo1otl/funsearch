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


def run_funsearch_process(formula: str, theory_explanation: str, constants_description: str, variables_description: str, data: str, file_upload: gr.File, insights: str,
                          max_nparams: int, max_mutations: int, request: gr.Request, auto_cleanup: bool, slack_checkbox):
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
    if slack_checkbox:
        try:
            notifier = slack.SlackNotifier()
        except ValueError:
            # Slack設定が無い場合は通知無しで続行
            pass

    start_time = time.time()

    if file_upload is not None:
        try:
            with open(file_upload.name, 'r', encoding='utf-8') as f:  # type: ignore
                data = f.read()
            full_log += \
                f"1. Loaded data from {file_upload.name}.\n"  # type: ignore
        except Exception as e:
            yield f"Error reading file: {e}\n{traceback.format_exc()}\n", UPDATE_HEADER
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
        args=(q, formula, theory_explanation, constants_description, variables_description, insights, inputs_np,
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


default_formula = r'E_composite = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)'
default_theory_explanation = r'''このモデルは、粒子で充填されたゴム複合材料の引張弾性率を予測することを目的としています。
基礎となる物理モデルは、複合材料の弾性率を定義するReussモデルです。'''
default_constants_description = r'''E_m: マトリックスの引張弾性率 (4.84で固定)
E_f: 充填材の引張弾性率 (117.64で固定)'''
default_variables_description = r'phi: フィラー体積分率 (実験データCSVの入力列)'
default_data = r'''0,4.84
0.09,5.56
0.17,6.13
0.33,10.13
0.44,14.96'''
default_insights = r'''進化の出発点は提供されたReussモデルです。
最大で MAX_NPARAMS 個の最適化可能なパラメータ（params 配列から）を導入して、Reussモデルを修正または拡張し、実験データとの適合性を向上させることを目指してください。
最終的な目標は、基本的なReussモデルに対して、物理的に意味のある改善を見つけ出すことです。'''
default_nparams = 1
default_max_mutations = 50

with gr.Blocks(theme=gr.themes.Soft()) as demo:  # type: ignore
    gr.Markdown("# FunSearch Gradio Interface (With Enhanced Stop Button)")

    with gr.Row():
        with gr.Column(scale=1):
            formula_input = gr.Textbox(
                lines=2, label="理論式", value=default_formula, info="進化の出発点となる数式を入力します。")
            theory_explanation_input = gr.Textbox(
                lines=3, label="理論式の説明", value=default_theory_explanation, info="数式の背景や目的を説明します。")
            constants_description_input = gr.Textbox(
                lines=3, label="定数の説明", value=default_constants_description, info="進化の過程で変更してはならない定数とその値を記述します。")
            variables_description_input = gr.Textbox(
                lines=2, label="説明変数の説明", value=default_variables_description, info="データCSVの入力列（目的変数の列を除く）に対応する変数を説明します。")
            data_input = gr.Textbox(
                lines=5, label="データ (CSV)", value=default_data, info="ファイルアップロード機能を使用する場合、これらのデータは無視されます。")
            file_upload = gr.File(
                label="またはCSVファイルをアップロード", file_types=[".csv"])
            insights_input = gr.Textbox(
                lines=3, label="着眼点", value=default_insights, info="進化の方向性をガイドするための追加のヒントや制約を記述します。")
            max_nparams_input = gr.Number(
                label="最大パラメータ数", value=default_nparams, precision=0, step=1,
                info="進化の過程で追加できる最大のパラメータ数を指定します。")
            max_mutations_input = gr.Number(
                label="変異回数", value=default_max_mutations, precision=0, step=1,
                info="回数に達するまで停止しないので必ず適切な値を設定してください。")
            auto_cleanup_checkbox = gr.Checkbox(
                label="ページ離脱時に自動停止", value=True,
                info="チェックを外すと、ページを離脱してもバックグラウンドで実行が継続されます。")
            slack_checkbox = gr.Checkbox(
                label="Slack通知", value=True,
                info="実行完了時にSlackに結果を通知します。"
            )

            with gr.Row():
                run_button = gr.Button("実行", variant="primary")
                stop_button = gr.Button("停止", variant="stop")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="実行ログ", lines=25, autoscroll=True, show_copy_button=True)
            update_output = gr.Markdown(label="更新ログ (Best Functions)")

    run_event = run_button.click(
        fn=run_funsearch_process,
        inputs=[formula_input, theory_explanation_input, constants_description_input, variables_description_input, data_input, file_upload,
                insights_input, max_nparams_input, max_mutations_input, auto_cleanup_checkbox, slack_checkbox],
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
    demo.launch(auth=auth_creds, server_name="0.0.0.0", server_port=7860)
