import os
import time
import gradio as gr
import threading
import queue
from typing import Dict, Any, List
import numpy as np
import traceback
import matplotlib.pyplot as plt
from funsearch import function, archipelago, cluster, presenter, slack, datadriven
from funsearch.presenter.plot_component import OneDimensionalPlotComponent
from funsearch.presenter.display import SessionDisplayManager
from funsearch.presenter.math_expression_generator import MathExpressionGenerator
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
        'auto_cleanup': auto_cleanup,
        'skeletons': [],
        'display_manager': SessionDisplayManager()
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

    # プロット用データを初期化（探索開始時）
    dataset = datadriven.Dataset(max_nparams, inputs_np, outputs_np)
    plot_component = OneDimensionalPlotComponent(dataset)
    sessions[session_hash]['plot_component'] = plot_component
    full_log += "--- Plot component initialized. ---\n"

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

    # 最終的なプロット状態の確認
    if session_hash in sessions:
        session_data = sessions[session_hash]
        plot_component = session_data.get('plot_component')
        if plot_component and 'skeletons' in session_data:
            full_log += f"--- Plot component ready. {len(session_data['skeletons'])} functions available for visualization. ---\n"

    yield full_log, UPDATE_HEADER + "".join(update_list)


def stop_funsearch_process(request: gr.Request):
    """FunSearchプロセスを停止"""
    session_hash = request.session_hash
    session_data = sessions[session_hash]  # type: ignore
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
    if session_hash not in sessions:
        return

    session_data = sessions[session_hash]
    if not session_data.get('auto_cleanup', True):
        return

    session_data['cancelled'] = True
    evolver = session_data.get('evolver')
    if evolver is not None:
        evolver.stop()
        session_data['evolver'] = None

    del sessions[session_hash]


# 可視化用の関数群
def update_selected_function_display(selected_functions, request: gr.Request):
    """選択された関数群の表示情報を更新"""
    if not selected_functions:
        return "関数を選択してください"

    session_hash = request.session_hash
    if session_hash not in sessions:
        return "セッションが見つかりません"

    session_data = sessions[session_hash]
    display_manager = session_data['display_manager']

    # 選択されたすべての関数の情報を収集
    selected_skeleton_infos = []

    # まず探索中のskeletonsから探す
    if 'skeletons' in session_data and session_data['skeletons']:
        for selected_idx in selected_functions:
            for skeleton_info in session_data['skeletons']:
                if skeleton_info['index'] == selected_idx:
                    selected_skeleton_infos.append(skeleton_info)
                    break

    # 探索完了後はplot_componentからも取得可能
    plot_component = get_plot_component(request)
    if plot_component:
        for selected_idx in selected_functions:
            if selected_idx in plot_component.skeletons:
                # 既に見つかっていない場合のみ追加
                if not any(info['index'] == selected_idx for info in selected_skeleton_infos):
                    skeleton_info_obj = plot_component.skeletons[selected_idx]
                    skeleton_info = {
                        'index': selected_idx,
                        'skeleton': skeleton_info_obj.skeleton,
                        'description': skeleton_info_obj.description,
                        'score': skeleton_info_obj.score
                    }
                    selected_skeleton_infos.append(skeleton_info)

    if selected_skeleton_infos:
        display_manager.set_selected_functions(selected_skeleton_infos)
        return display_manager.get_current_markdown()

    return "選択された関数が見つかりません"


def generate_math_expression(selected_functions, request: gr.Request):
    """選択されたすべての関数に対して数式表現を一括生成"""
    if not selected_functions:
        return "関数を選択してください"

    session_hash = request.session_hash
    if session_hash not in sessions:
        return "セッションが見つかりません"

    session_data = sessions[session_hash]
    display_manager = session_data['display_manager']

    if not display_manager.has_selected_functions():
        return "まず関数を選択してください"

    # 数式表現がまだない関数のみを取得
    functions_needing_math = display_manager.get_functions_needing_math()

    if not functions_needing_math:
        return display_manager.get_current_markdown()  # 全ての関数に既に数式がある

    # 数式生成が必要な関数の情報を収集
    skeleton_infos = []
    for idx in functions_needing_math:
        if idx in display_manager.all_functions:
            function_info = display_manager.all_functions[idx]
            skeleton_infos.append(function_info.skeleton_info)

    if not skeleton_infos:
        return display_manager.get_current_markdown()

    # 実際のGeneratorで数式表現を生成
    generator = MathExpressionGenerator(GEMINI_CLIENT_FOR_CONVERTER)
    skeletons = [info['skeleton'] for info in skeleton_infos]
    expressions = generator.generate_expressions(skeletons)

    # indexと対応付けて辞書に変換
    expressions_dict = {}
    for i, info in enumerate(skeleton_infos):
        if i < len(expressions):
            expressions_dict[info['index']] = expressions[i]

    # 一括追加
    display_manager.add_math_expressions(expressions_dict)

    return display_manager.get_current_markdown()


def get_plot_component(request: gr.Request):
    """セッションからplot_componentを取得"""
    session_hash = request.session_hash
    if session_hash not in sessions:
        return None
    session_data = sessions[session_hash]
    return session_data.get('plot_component')


def get_available_functions(request: gr.Request):
    """現在のセッションの関数リストを返す"""
    session_hash = request.session_hash
    if session_hash not in sessions:
        return gr.CheckboxGroup(choices=[], label="関数選択", info="実行開始後に関数が表示されます")

    session_data = sessions[session_hash]

    # 探索中でもskeletonsから関数リストを取得
    if 'skeletons' in session_data and session_data['skeletons']:
        choices = [(f"{skeleton_info['description']}", skeleton_info['index'])
                   for skeleton_info in session_data['skeletons']]
        return gr.CheckboxGroup(choices=choices, label="関数選択")

    # 探索完了後はplot_componentからも取得可能
    plot_component = get_plot_component(request)
    if plot_component:
        functions = plot_component.get_available_skeletons()
        choices = [(f"{desc}", idx) for idx, desc, score in functions]
        return gr.CheckboxGroup(choices=choices, label="関数選択")

    return gr.CheckboxGroup(choices=[], label="関数選択", info="関数が見つかるまでお待ちください")


def create_empty_plot(message):
    """空のプロットを作成"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center',
            va='center', transform=ax.transAxes)
    return fig


def create_plot(selected_functions, request: gr.Request):
    """選択された関数でプロットを作成"""
    if not selected_functions:
        return create_empty_plot('関数を選択してください'), ""

    # 選択された関数の表示情報を更新
    selected_display = update_selected_function_display(
        selected_functions, request)

    plot_component = get_plot_component(request)
    if not plot_component:
        return create_empty_plot('プロットコンポーネントが初期化されていません'), selected_display

    plot_component.select_skeletons(selected_functions)
    plot_data = plot_component.create_plot_data()

    from funsearch.presenter.plot_component import create_matplotlib_plot
    plot = create_matplotlib_plot(plot_data)

    return plot, selected_display


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

    with gr.Tabs() as tabs:
        with gr.TabItem("🚀 実行"):
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
                    update_output = gr.Markdown(
                        "## Best Functions Found"
                    )

        with gr.TabItem("📊 可視化") as viz_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_functions_btn = gr.Button(
                        "関数リスト更新", variant="secondary")
                    function_selection = gr.CheckboxGroup(
                        choices=[], label="関数選択", info="実行完了後に関数が表示されます")
                    generate_math_btn = gr.Button(
                        "数式表現を生成", variant="secondary")

                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="関数比較プロット")
                    code_output = gr.Markdown(
                        label="選択された関数の詳細",
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": True},
                            {"left": "$$", "right": "$$", "display": True}
                        ]
                    )

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

    # 可視化タブのイベント
    refresh_functions_btn.click(
        fn=get_available_functions,
        outputs=function_selection
    )

    function_selection.select(
        fn=create_plot,
        inputs=[function_selection],
        outputs=[plot_output, code_output]
    )

    generate_math_btn.click(
        fn=generate_math_expression,
        inputs=[function_selection],
        outputs=code_output
    )

    demo.unload(
        fn=cleanup_session,
    )

if __name__ == "__main__":
    print("Launching Gradio UI...")
    auth_creds = None
    demo.launch(auth=auth_creds, server_name="0.0.0.0", server_port=7860)
