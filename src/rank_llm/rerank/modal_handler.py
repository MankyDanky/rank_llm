class ModalHandler:
    def __init__(self, model_path: str, batch_size: int):
        try:
            import modal
        except ImportError as error:
            raise ImportError(
                "Install rank-llm[modal] to use Modal inference"
            ) from error

        model = modal.Cls.from_name("gaggle-rerank", "RankLLMModel")
        self._model = model(model_path=model_path, batch_size=batch_size)

    def run_llm_batched(self, prompts):
        return self._model.run_llm_batched.remote(prompts)
