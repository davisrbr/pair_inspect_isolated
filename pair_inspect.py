from inspect_ai import Task, Epochs, mean, stderr

from data_inspect import jb_behaviors_dataset
from solvers_inspect import pair_solver
from scorers_inspect import final_scorer


def pair_task(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    max_iterations: int = 3,
    n_last_messages: int = 2, 
    streams: int = 2, 
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    dataset: list = jb_behaviors_dataset,
    cutoff: int = 1,
):
    """
    PAIR task within Inspect

    Args:
        target_model_name (str): Name of the target language model.
        judge_model_name (str): Name of the judge language model.
        attack_model_name (str): Name of the attack language model.
        max_iterations (int): Maximum number of iterations to run PAIR.
        n_last_messages (int): Number of last messages to pass in as context for attacker model.
        streams (int): Number of PAIR streams to run.
        use_strongreject_scorer (bool): Whether to use the strongreject scorer.
        heirarchal_scorer (bool): Whether to use a heirarchal scorer. First uses weak model to judge, if score is above cutoff, uses strong model to judge.
        dataset (list): Dataset to use for the task.
        cutoff (int): Cutoff score (between 0 and 1) to qualify for jailbreakbench.
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
        scorer=final_scorer(cutoff=cutoff),
        metrics=[mean(), stderr()],
        epochs=Epochs(streams, reducer="max")
    )
