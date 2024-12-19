from inspect_ai import Task, task, Epochs, eval
from data_inspect import jb_behaviors_dataset
from solvers_inspect import pair_solver
from scorers_inspect import final_scorer

@task
def pair_task(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    max_iterations: int = 3,
    n_last_messages: int = 2, 
    epochs: int = 2, 
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    dataset: list = jb_behaviors_dataset,
):
    """
    PAIR task within Inspect
    """
    return Task(
        dataset=dataset,
        plan=[
            pair_solver(
                max_iterations=max_iterations,
                target_model_name=target_model_name,
                judge_model_name=judge_model_name,
                attack_model_name=attack_model_name,
                n_last_messages=n_last_messages,
                use_strongreject_scorer=use_strongreject_scorer,
                heirarchal_scorer=heirarchal_scorer,
            ),
        ],
        scorer=final_scorer(),
        epochs=Epochs(epochs)
    )
