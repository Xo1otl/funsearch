# C4 Model

## Level 1: System Context Diagram (システムコンテキスト図)

GASシステム、ユーザー（研究者）、主要な外部システムとの関係性を示す。

```mermaid
C4Context
    title Level 1: System Context Diagram for GAS (Generative Ansatz Search)

    Person(researcher, "研究者", "化学・物理モデルを発見するユーザー")
    System(gas, "GAS Framework", "数式・理論モデル発見ライブラリ")
    System_Ext(llm, "LLM", "仮説生成・評価に用いる外部基盤モデル")
    System_Ext(external_analysis, "外部分析基盤", "分析・データ保存用プラットフォーム (任意)")

    Rel(researcher, gas, "探索実行・結果分析")
    Rel(gas, llm, "仮説生成・評価をAPI依頼")
    Rel(gas, external_analysis, "分析データを転送")

    UpdateLayoutConfig($c4ShapeInRow="1")
```

## Level 2: Container Diagram (コンテナ図)

GASライブラリを構成する4つのサービスと2つのデータストアをコンテナとして示す。

```mermaid
C4Container
    title Level 2: Container Diagram for GAS

    Person(researcher, "研究者")
    System_Ext(llm, "LLM")
    System_Ext(external_analysis, "外部分析基盤 (任意)")

    System_Boundary(gas, "GAS Framework") {
        Container(ui_service, "IV. UI Service", "Web App/CLI", "ユーザーインターフェース")
        Container(command_service, "I. Command Service", "Backend/Worker", "探索プロセスの実行")
        Container(projection_service, "II. Projection Service", "Data Processor", "状態データの変換・転送")
        Container(query_service, "III. Query Service", "API Service", "分析データへのクエリ実行")

        ContainerDb(primary_db, "Primary Datastore", "RDB/NoSQL", "探索プロセスの状態を永続化")
        ContainerDb(analysis_db, "Analysis Datastore", "DWH/RDB", "分析・可視化用に最適化されたデータ")
    }

    Rel(researcher, ui_service, "利用", "HTTPS/CLI")
    Rel(ui_service, command_service, "探索リクエスト", "API/Queue")
    UpdateRelStyle(ui_service, command_service, $offsetY="-40")
    Rel(ui_service, query_service, "結果照会", "API")
    UpdateRelStyle(ui_service, query_service, $offsetY="30", $offsetX="60")

    Rel(command_service, primary_db, "R/W", "状態の永続化・復元")
    UpdateRelStyle(command_service, primary_db, $offsetY="40")
    Rel(command_service, llm, "APIリクエスト", "仮説生成・評価")
    UpdateRelStyle(command_service, llm, $offsetX="-60", $offsetY="-20")

    Rel(projection_service, primary_db, "Read", "状態データ取得")
    UpdateRelStyle(projection_service, primary_db, $offsetX="-35", $offsetY="40")
    Rel(projection_service, analysis_db, "Write", "変換データ保存")
    Rel(projection_service, external_analysis, "データ転送")
    UpdateRelStyle(projection_service, external_analysis, $offsetY="-30")

    Rel(query_service, analysis_db, "Query", "データ照会")
    UpdateRelStyle(query_service, analysis_db, $offsetY="35")

    UpdateLayoutConfig($c4ShapeInRow="3")
```

## Level 3: Component Diagram for Command Service

`Command Service`の内部コンポーネントと、`Orchestrator`が管理する状態遷移フローを示す。

```mermaid
C4Component
    title Level 3: Component Diagram for Command Service

    Container(ui_service, "IV. UI Service")
    ContainerDb(primary_db, "Primary Datastore")
    System_Ext(external_services, "External Services", "LLM, Solvers etc.")

    Container_Boundary(command_service, "I. Command Service") {

        Component(controller, "Controller", "ライフサイクル管理")
        Component(repository, "Repository", "永続化層 (Caretaker)")
        
        Component(orchestrator, "Orchestrator", "実行エンジン")
        
        Component(propose_fn, "ProposeFn", "仮説生成器 (Stateless)")
        Component(observe_fn, "ObserveFn", "仮説評価器 (Stateless)")
        Component(propagate_fn, "PropagateFn", "更新戦略 (Stateless)")

        Component(search_state, "SearchState", "探索状態 (Originator)")
    }
    
    %% --- Setup & Persistence Flow ---
    Rel(ui_service, controller, "1. 探索要求", "API Call")
    Rel(controller, repository, "2. 状態の永続化/復元を指示")
    Rel(repository, primary_db, "R/W", "DB Transaction")
    Rel(repository, search_state, "Mementoで状態をGet/Set")
    Rel(controller, orchestrator, "3. 探索開始を指示")

    %% --- Core Execution Loop ---
    Rel(orchestrator, propose_fn, "a. (Query, Context)生成を指示")
    Rel(propose_fn, search_state, "参照")

    Rel(orchestrator, observe_fn, "b. Evidence生成を指示", "(Queryを渡す)")
    Rel(observe_fn, external_services, "利用 (任意)")

    Rel(orchestrator, propagate_fn, "c. 新SearchState計算を指示", "propagate_fn(query, context, evidence, search_state)")
    Rel(propagate_fn, search_state, "参照")

    Rel(orchestrator, search_state, "d. 新状態で更新")
    
    UpdateLayoutConfig($c4ShapeInRow="3")
```

# SequenceDiagram
```mermaid
sequenceDiagram
    participant App as Application
    participant Orch as Orchestrator
    participant pChan as ProposeReqs(Ch)
    participant pResChan as ProposeResults(Ch)
    participant pWorkers as ProposeWorker Pool
    participant oChan as ObserveReqs(Ch)
    participant oResChan as ObserveResults(Ch)
    participant oWorkers as ObserveWorker Pool

    rect rgba(13, 0, 88, 1)
        App->>pChan: Creates Channel
        App->>pResChan: Creates Channel
        App->>oChan: Creates Channel
        App->>oResChan: Creates Channel
        App->>pWorkers: go runProposeWorker(pChan, pResChan)
        App->>oWorkers: go runObserveWorker(oChan, oResChan)
        App->>Orch: NewOrchestrator(channels...)
        App->>Orch: go orchestrator.Run(initialState)
    end

    activate Orch
    Note over Orch: Search begins. Dispatch initial propose tasks.
    Orch->>pChan: ProposeRequest
    Orch->>pChan: ProposeRequest
    deactivate Orch

    pWorkers->>pChan: Receives a request
    pWorkers->>pResChan: ProposeResult { Query_A, ProposeContext_A }

    activate Orch
    Orch->>pResChan: Receives ProposeResult
    Note right of Orch: ProposeResult received.<br/>Generate new CorrelationID "A".<br/>Store Query_A, ProposeContext_A internally using "A".<br/>Dispatch observe task with CorrelationID "A".
    Orch->>oChan: ObserveRequest { Context: {CorrelationID:"A"}, Query_A }
    deactivate Orch

    pWorkers->>pChan: Receives another request
    pWorkers->>pResChan: ProposeResult { Query_B, ProposeContext_B }
    
    activate Orch
    Orch->>pResChan: Receives ProposeResult
    Note right of Orch: Another ProposeResult received.<br/>Generate new CorrelationID "B".<br/>Dispatch observe task with CorrelationID "B".<br/>Orchestrator now waits for next event.
    Orch->>oChan: ObserveRequest { Context: {CorrelationID:"B"}, Query_B }
    deactivate Orch

    %% Observe results can arrive in any order %%
    oWorkers->>oChan: Receives Request "B"
    oWorkers->>oResChan: ObserveResult { Context: {CorrelationID:"B"}, Evidence_B }
    
    activate Orch
    Orch->>oResChan: Receives ObserveResult "B"
    Note right of Orch: Observe Result "B" arrives first.<br/>Retrieve context for "B", update state.<br/>Dispatch a new propose task.
    Orch->>pChan: ProposeRequest
    deactivate Orch

    oWorkers->>oChan: Receives Request "A"
    oWorkers->>oResChan: ObserveResult { Context: {CorrelationID:"A"}, Evidence_A }

    activate Orch
    Orch->>oResChan: Receives ObserveResult "A"
    Note right of Orch: Observe Result "A" arrives.<br/>Update state using its context.<br/>Dispatch another new propose task.
    Orch->>pChan: ProposeRequest
    deactivate Orch
```
