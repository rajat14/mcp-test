from typing import Dict
from src.core.config.config_refs import ConfigNodes
from src.core.config.nemo_config.nemo_config import Rails
from src.core.instrumentation.logger.logger import Logger
from src.core.instrumentation.decorators.function_timer import timed
from src.core.instrumentation.decorators.log_entry_exit import log_entry
from src.infra.llm_connectors.connector_factory import LlmConnectorFactory

logger = Logger()


class InferenceService:

    def __init__(self):
        pass

    @timed()
    @log_entry()
    def query_via_config(
            self,
            llm_connector_name: str,
            prompt: str,
            max_tokens_override: int = None,
            inference_params_override: Dict = None
    ) -> str:
        """
        Query the specified llm with a fully constructed prompt. Guardrails will validate the prompt and/or output if
        Guardrails is enabled in the config.

        :param llm_connector_name: Which llm connector to use. A connector with this name must be present in the config.
        :param prompt: Prompt to pass to the LLM (and optionally to Guardrails).
        :param max_tokens_override: If not None, this will override the inference param's max tokens setting.
        :param inference_params_override: If not None, this will override the entire config inference params dict.
        :return:
        """
        inference_params = self.__get_inference_params(
            llm_connector_name, max_tokens_override, inference_params_override
        )
        connector = LlmConnectorFactory.get_connector_via_config(llm_connector_name)

        if Rails().guardrails_is_enabled(llm_connector_name):
            logger.info("Applying Guardrails validation")
            prompt, response = Rails().process_prompt_with_guardrails(
                llm_connector_name=llm_connector_name,
                query=prompt,
                query_context_data=None
            )

        else:
            response = connector.query(prompt=prompt, inference_params=inference_params)

        logger.info(f"Inference params: {inference_params}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")

        return response

    def __get_inference_params(
            self,
            llm_connector_name: str,
            max_tokens_override: int = None,
            inference_params_override: Dict = None
    ) -> Dict:
        llm_config = ConfigNodes.get_llm_config(llm_connector_name)

        if inference_params_override is None:
            inference_params = llm_config.base_inference_parameters
        else:
            inference_params = inference_params_override

        if max_tokens_override is not None:
            if llm_config.base_inference_parameters.get("max_new_tokens") is not None:
                max_tokens_config_name = "max_new_tokens"
            else:
                max_tokens_config_name = "max_tokens"
            inference_params[max_tokens_config_name] = max_tokens_override

        return inference_params
