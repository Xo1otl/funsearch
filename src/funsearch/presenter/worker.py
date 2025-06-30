import queue
import traceback
import time
from typing import Dict, Any, Optional
import numpy as np
from funsearch import llmsr, datadriven
from . import CancellableInputConverter, SessionQueueProfiler
from .domain import ResultNotifier
from funsearch.function.py_ast_skeleton import PyAstSkeleton


def funsearch_worker(q: queue.Queue, formula: str, theory_explanation: str, constants_description: str, variables_description: str, insights: str,
                     inputs: np.ndarray, outputs: np.ndarray, max_nparams: int,
                     max_mutations: int, session_data: Dict[str, Any], gemini_client,
                     notifier: Optional[ResultNotifier] = None, start_time: Optional[float] = None):
    evolver = None
    top_functions = []

    try:
        if session_data.get('cancelled', False):
            q.put(('log', "--- Process cancelled before starting. ---\n"))
            return

        q.put(('log', "--- Starting FunSearch Worker ---\n"))

        converter = CancellableInputConverter(
            gemini_client, session_data)
        q.put(('log', "2. Calling LLM to convert input...\n"))

        info = converter.convert(
            formula, theory_explanation, constants_description, variables_description, insights)

        if not info or not info.get("equation_src"):
            q.put(('log', "| Error | InputConverter failed or returned empty source.\n"))
            return

        if session_data.get('cancelled', False):
            q.put(('log', "--- Process cancelled after conversion. ---\n"))
            return

        q.put(
            ('log', f"--- Generated Code ---\n{info['equation_src']}\n---\n"))

        # 初期Skeletonを保存
        try:
            initial_skeleton = PyAstSkeleton(info["equation_src"])

            initial_skeleton_info = {
                'index': 0,
                'skeleton': initial_skeleton,
                'score': 'Initial',
                'code': info["equation_src"],
                'description': f"Initial Function (Generated)"
            }
            session_data['skeletons'].append(initial_skeleton_info)
            q.put(('log', f"--- Added initial function to session data ---\n"))
        except Exception as e:
            q.put(
                ('log', f"--- Warning: Could not save initial function: {e} ---\n"))

        # 通知用に関数を収集するためのカスタムプロファイラ
        class NotificationProfiler(SessionQueueProfiler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def profile(self, event):
                super().profile(event)
                if event.type == "on_best_island_improved":
                    try:
                        best_fn = event.payload.best_fn()
                        score = self._get_score(best_fn)
                        code = self._format_function(best_fn)
                        skeleton = best_fn.skeleton()

                        if 'skeletons' not in session_data:
                            session_data['skeletons'] = []

                        skeleton_info = {
                            'index': len(session_data['skeletons']),
                            'skeleton': skeleton,
                            'score': score,
                            'code': code,
                            'description': f"Evolved Function {len(session_data['skeletons'])} (Score: {score})"
                        }
                        session_data['skeletons'].append(skeleton_info)

                        top_functions.append(
                            f"**Score: {score}**\n```python\n{code}\n```")
                        if len(top_functions) > 10:
                            top_functions.pop(0)

                    except Exception:
                        pass

        profiler = NotificationProfiler(q, max_mutations, session_data)

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

        session_data['evolver'] = evolver

        if session_data.get('cancelled', False):
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
        if 'evolver' in session_data:
            session_data['evolver'] = None

        if notifier and top_functions:
            try:
                if start_time:
                    start_str = time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(start_time))
                    end_str = time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                else:
                    start_str = end_str = "Unknown"

                header_message = f"""🔬 FunSearch Completed

📊 *Execution Summary:*
• Max Parameters: {max_nparams}
• Max Mutations: {max_mutations}
• Start Time: {start_str}
• End Time: {end_str}

🧪 *Formula Text:*
{formula}

🔬 *Theory Explanation:*
{theory_explanation}

🔢 *Constants Description:*
{constants_description}

📝 *Variables Description:*
{variables_description}

💡 *Insights:*
{insights}

🏆 *Top Functions Found ({len(top_functions)}):*"""
                top_functions.reverse()
                messages = [header_message] + top_functions
                success = notifier.send_message(messages)
                q.put(('log', f"✅ Slack notification sent: {success}\n"))
            except Exception as e:
                q.put(('log', f"❌ Slack notification error: {e}\n"))

        q.put(('end', None))
