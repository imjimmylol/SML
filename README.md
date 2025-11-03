# SML-2 Project

## Environment Data Flow

This diagram illustrates the data flow within the simulation environment, particularly how two parallel economies are created and managed from a single state object.

```mermaid
graph TD
    subgraph Initialization
        A["main()"] -- reads --> B("config.yaml");
        B -- parameters --> C["EconPackEnv"];
        C -- "is wrapped by" --> D["EnvBuffer"];
        D -- calls --> E["env.reset()"];
    end

    subgraph "State Creation"
        E --> F{"Generate Base State"};
        F --> G["Sample money, savings"];
        F --> H["Sample v0"];
        H -- normalized --> I["v0_norm"];
        I -- ".clone()" --> J["v1"];
        I -- ".clone()" --> K["v2"];
        J & K & G --> L["Initial StateDict"];
        L --> M{"state contains<br/>v1, v2, etc."};
        M --> D;
    end
    
    subgraph "Simulation Loop"
        N["Loop"] --> O{"Get Observations"};
        O -- "branch='v1'" --> P["buffer.get_obs('v1')"];
        O -- "branch='v2'" --> Q["buffer.get_obs('v2')"];
        
        P --> P1["env.get_obs() for v1"];
        Q --> Q1["env.get_obs() for v2"];

        P1 --> R["env._build_inputs()"];
        Q1 --> S["env._build_inputs()"];

        R --> T["Obs_v1"];
        S --> U["Obs_v2"];

        T & U -- "fed into" --> V["model"];
        V --> W["actions_v1, actions_v2"];

        W --> X["actions dictionary"];
        X -- "passed to" --> Y["buffer.step()"];
    end

    subgraph "State Transition"
        Y --> Z["demo_transition()"];
        Z --> AA{"Update Logic"};
        AA --> AB["v1_next"];
        AA --> AC["v2_next"];
        AA --> AD["shared_vars_next"];
        AB & AC & AD --> AE["next_state"];
        AE -- "returned to" --> Y;
        Y --> AF["Update buffer.state"];
        AF --> N;
    end

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#f9f,stroke:#333,stroke-width:2px
    style AE fill:#f9f,stroke:#333,stroke-width:2px
    style T fill:#ccf,stroke:#333,stroke-width:1px
    style U fill:#ccf,stroke:#333,stroke-width:1px
```

### Explanation of the Flow

1.  **Initialization (`main.py`)**:
    *   The `main()` function loads a configuration.
    *   It creates an `EconPackEnv` instance, which knows how to create initial states and format them into observations.
    *   This `env` is then wrapped by `EnvBuffer`, which holds the state and manages simulation steps.
    *   `EnvBuffer` immediately calls `env.reset()` to create the initial `state`.

2.  **State Creation (`EconPackEnv.reset`)**:
    *   This is where the parallel economies are born.
    *   It samples a *single* base tensor `v0`.
    *   **The Key Step**: It creates `v1 = v0.clone()` and `v2 = v0.clone()`. By using `.clone()`, it ensures `v1` and `v2` start with identical values but are two separate tensors. Future modifications to one will not affect the other.
    *   The initial `state` dictionary, containing both `v1` and `v2`, is returned.

3.  **Simulation Loop (`main.py`)**:
    *   At each step `t`:
    *   **Observation Generation**: `buffer.get_obs()` is called twice, once for each "branch" (`v1` and `v2`), creating two sets of observations for the model.
    *   **Model Inference**: The model processes both sets of observations to produce actions for each parallel economy.
    *   **State Transition**: `buffer.step()` is called, passing the actions and the `demo_transition` function.

4.  **State Transition (`EnvBuffer.step` & `demo_transition`)**:
    *   `demo_transition` receives the current state and actions.
    *   It calculates `v1_next` and `v2_next` *independently* using their respective growth rates.
    *   It calculates the next values for shared variables (e.g., `money_next`).
    *   It packs everything into a `next_state` dictionary, which is used to update the buffer's state for the next iteration.


## Labor FOC loss

TAX_PARAMS = {
    "tax_consumption": 0.065,          # Consumption tax (fixed)
    "tax_income": 0.2,                # Tax on labor income
    "income_tax_elasticity": 0.5,     # Elasticity of labor supply w.r.t. after-tax income
    "saving_tax_elasticity": 0.5,     # Elasticity of savings w.r.t. after-tax income
    "tax_saving": 0.1                 # Tax on interest income
}

converge


TAX_PARAMS = {
    "tax_consumption": 0.065,          # Consumption tax (fixed)
    "tax_income": 0.5,                # Tax on labor income
    "income_tax_elasticity": 0.5,     # Elasticity of labor supply w.r.t. after-tax income
    "saving_tax_elasticity": 0.5,     # Elasticity of savings w.r.t. after-tax income
    "tax_saving": 0.5                 # Tax on interest income
}

diverge for gamma = 2 or gamma = 0.5