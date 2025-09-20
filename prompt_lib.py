import os
from typing import Dict, List
from src.core.util.url_util import UrlUtil
from src.infra.storage_connectors.connectors.local_storage_connector import LocalStorageConnector
from src.infra.storage_connectors.file_support.loaded_file import LoadedFile
from src.domain.agents.base_prompt_library import BasePromptLibrary


class LocalFSPromptLibrary(BasePromptLibrary):

    prompts: Dict[str, str] = None

    def __init__(
            self,
            local_storage_connector: LocalStorageConnector,
            default_prompt_path_for_lookups: str
    ):
        """
        Init a local filesystem prompt library. This will load all the prompts that the local storage connector is
        pointing to and store them in a dict.

        For example, let's say local_storage_connector.base_file_path = "/tmp/prompt_path/", and that folder contains
        these prompts:
          /tmp/prompt_path/nested_path/prompt_1.txt
          /tmp/prompt_path/nested_path/prompt_2.txt

        The prompts will then be stored in a dict as such:
        {
            "nested_path/prompt_1.txt": "This is the first question.",
            "nested_path/prompt_2.txt": "This is the second question."
        }

        If default_prompt_path_for_lookups = "nested_path/", then the retrieve_prompt_via_defaults method will use
        "nested_path/" as the base part of the dict key while retrieving prompts.

        :param local_storage_connector: Local storage connector which can load the prompts.
        :param default_prompt_path_for_lookups: Default base path to use with retrieve_prompt_via_defaults.
        """
        super().__init__(
            default_prompt_path_for_lookups=default_prompt_path_for_lookups
        )

        self.local_storage_connector = local_storage_connector

        # Prompts are a static var so that they do not need to be reloaded each time
        if LocalFSPromptLibrary.prompts is None:
            LocalFSPromptLibrary.prompts = self.load_all_prompts()

    def load_all_prompts(
            self
    ) -> Dict[str, str]:
        loaded_files: List[LoadedFile] = self.local_storage_connector.load_base_file_path_files_recursively()

        prompts = {}
        for loaded_file in loaded_files:
            # All prompt keys use forward-slashes so that they remain consistent across platforms and library types
            prompt_file_relative_path = loaded_file.path_after_base_file_path
            prompt_file_relative_path = prompt_file_relative_path.replace(os.sep, "/")
            prompt_key = prompt_file_relative_path
            prompts[prompt_key] = loaded_file.file_contents

        return prompts

    def retrieve_prompt(
            self,
            prompt_key: str,
            placeholder_names_and_values: Dict = None
    ) -> str:
        if placeholder_names_and_values is None:
            placeholder_names_and_values = {}

        if prompt_key not in self.prompts:
            raise Exception(f"The prompt key {prompt_key} was not found in the prompt library; available keys: {self.prompts.keys()}")

        prompt = self.prompts[prompt_key]
        prompt = prompt.format(**placeholder_names_and_values)
        return prompt

    def retrieve_prompt_via_defaults(
            self,
            prompt_name: str = "prompt",
            placeholder_names_and_values: Dict = None
    ) -> str:
        # All prompt keys use forward-slashes so that they remain consistent across platforms and library types
        prompt_key = UrlUtil.construct_url([self.default_prompt_path_for_lookups, f"{prompt_name}.txt"])
        prompt = self.retrieve_prompt(prompt_key, placeholder_names_and_values)
        return prompt

    def retrieve_prompt_via_llm_connector(
            self,
            llm_connector_name: str,
            prompt_name: str = "prompt",
            placeholder_names_and_values: Dict = None
    ) -> str:
        # All prompt keys use forward-slashes so that they remain consistent across platforms and library types
        prompt_key = UrlUtil.construct_url([self.default_prompt_path_for_lookups, llm_connector_name, f"{prompt_name}.txt"])
        prompt = self.retrieve_prompt(prompt_key, placeholder_names_and_values)
        return prompt
