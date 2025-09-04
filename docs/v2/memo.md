# 現在のフレームワークにおけるorchestrateの注意点
非同期島モデルを行う場合queueを中心とした実装になる。

生成→評価→更新の流れや型は変える必要はないが、全体をループするworkerを横に並列化する必要がある。

要するに上位存在が一つ増える。

orchestrator, worker(New), propose, observe, propagateが必要で、現在の設計モデルはworkerが欠けてる。

# プログラムスケルトン
```python
import asyncio
from typing import List

# --- 型定義 (省略) ---

# ===================================================================
# 1. IslandWorker: 個々の島を進化させるロジック
# ===================================================================
async def island_worker(
    initial_state: IslandState,
    inbox: asyncio.Queue,         # 司令受信用
    status_queue: asyncio.Queue   # 状態報告用
):
    """単一の島を進化させ続ける、自律したワーカー"""
    state = initial_state
    
    # 島ごとの propose/observe/propagate 関数 (ここでは仮定義)
    propose = ...
    observe = ...
    propagate = ...

    while state.status != "terminated":
        # コーディネーターからの司令をチェック
        if not inbox.empty():
            command = await inbox.get()
            # ...司令に応じた処理（リセット、終了など）...
            if command["type"] == "terminate":
                break

        # 島単位の探索サイクル
        query, context = await propose(state)
        evidence = await observe(query)
        state = propagate(state, query, context, evidence)

        # 自身の状態を定期的に報告
        await status_queue.put({"id": state.id, "score": state.best_score})

# ===================================================================
# 2. GlobalCoordinator: 全体を監視・指揮するロジック
# ===================================================================
async def global_coordinator(
    mailboxes: Dict[int, asyncio.Queue],
    status_queue: asyncio.Queue,
    global_termination_strategy: Callable[..., bool]
):
    """全島の状態を監視し、戦略的な司令を出す管理棟"""
    all_statuses = {}

    while not global_termination_strategy(all_statuses):
        # 監視インターバル
        await asyncio.sleep(10)

        # 全島からの報告を収集
        while not status_queue.empty():
            report = await status_queue.get()
            all_statuses[report["id"]] = report
        
        # グローバルな判断と司令の発行
        # (例: スコアの悪い島を特定し、リセット命令を送る)
        # ...

    # 探索終了の司令を全島に送る
    for island_id in mailboxes:
        await mailboxes[island_id].put({"type": "terminate"})


# ===================================================================
# 3. AsyncSearchController: 全体を起動するエントリーポイント
# ===================================================================
async def run_asynchronous_search(
    initial_states: List[IslandState],
    global_termination_strategy,
    # ... 他の設定 ...
):
    """非同期島モデルの探索全体を管理・実行する"""
    
    # 通信チャネルの準備
    mailboxes = {state.id: asyncio.Queue() for state in initial_states}
    status_queue = asyncio.Queue()

    # 全IslandWorkerをタスクとして起動
    island_tasks = [
        asyncio.create_task(island_worker(state, mailboxes[state.id], status_queue))
        for state in initial_states
    ]

    # GlobalCoordinatorをタスクとして起動
    coordinator_task = asyncio.create_task(
        global_coordinator(mailboxes, status_queue, global_termination_strategy)
    )

    # 全タスクの完了を待つ
    await asyncio.gather(*island_tasks, coordinator_task)
    print("Asynchronous search has been completed.")
```