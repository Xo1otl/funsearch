from typing import List, Any, Optional
from datetime import datetime
from .domain import ResultNotifier


class FunSearchResult:
    """FunSearch実行結果を表すデータクラス"""
    
    def __init__(self, formula: str, params: str, insights: str, 
                 max_nparams: int, max_mutations: int):
        self.formula = formula
        self.params = params
        self.insights = insights
        self.max_nparams = max_nparams
        self.max_mutations = max_mutations
        self.top_functions: List[tuple] = []  # (score, function_code) のリスト
        self.evaluation_count = 0
        self.mutation_count = 0
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
    
    def add_function(self, score: Any, function_code: str):
        """関数を追加してトップ10を維持"""
        self.top_functions.append((score, function_code))
        # スコアでソート（降順）してトップ10を保持
        self.top_functions.sort(key=lambda x: str(x[0]), reverse=True)
        self.top_functions = self.top_functions[:10]
    
    def set_counters(self, evaluation_count: int, mutation_count: int):
        """カウンターを設定"""
        self.evaluation_count = evaluation_count
        self.mutation_count = mutation_count
    
    def finish(self):
        """実行終了時間を設定"""
        self.end_time = datetime.now()


def format_funsearch_notification(result: FunSearchResult) -> str:
    """FunSearch結果を通知用文字列にフォーマット"""
    duration = ""
    if result.end_time:
        delta = result.end_time - result.start_time
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    message = f"""🔬 FunSearch Completed

📊 **Execution Summary:**
• Formula: {result.formula[:100]}{'...' if len(result.formula) > 100 else ''}
• Parameters: {result.params}
• Max Parameters: {result.max_nparams}
• Max Mutations: {result.max_mutations}
• Evaluations: {result.evaluation_count}
• Mutations: {result.mutation_count}
• Duration: {duration}

💡 **Insights:**
{result.insights[:200]}{'...' if len(result.insights) > 200 else ''}

🏆 **Top Functions Found ({len(result.top_functions)}):**"""

    for i, (score, func_code) in enumerate(result.top_functions, 1):
        # 関数コードを短縮
        lines = func_code.split('\n')
        if len(lines) > 5:
            short_code = '\n'.join(lines[:3]) + '\n    ...\n' + lines[-1]
        else:
            short_code = func_code
        
        message += f"""

**#{i} - Score: {score}**
```python
{short_code[:300]}{'...' if len(short_code) > 300 else ''}
```"""

    return message


def send_funsearch_notification(result: FunSearchResult, notifier: ResultNotifier) -> bool:
    """FunSearch結果の通知を送信"""
    message = format_funsearch_notification(result)
    return notifier.send_message(message)